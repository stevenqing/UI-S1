import argparse
import json
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
CLOSING_DIR = ROOT / "runs/closing/2026-08-02"
sys.path.insert(0, str(H1_DIR))
sys.path.insert(0, str(H3_DIR))
sys.path.insert(0, str(CLOSING_DIR))
from aggregators_coord import mvp_official
from h3_eval import ccm_select, fit_ccm, point_in_bbox
from f1_paired_bootstrap import paired_bootstrap
from cala_common import (
    BUDGETS,
    MODEL_ORDER,
    SHARED_ACTIONS,
    UNIFORM_SEQUENCE,
    V_ONLY_SEQUENCE,
    action_name,
    build_rows,
    correctness,
    load_bank,
    mean_failure_kappa,
    sha256_file,
    split_ids,
)


SEED = 20260803
MDE = 0.007043345177520599


def b3_accuracy(context, row_ids, actions):
    correct = 0
    for row_id in row_ids:
        candidates = [context["bank"][action][row_id] for action in actions]
        points = [candidate["point"] for candidate in candidates]
        pseudo = [{"coverage": candidate.get("coverage", 0), "region": candidate["region"]} for candidate in candidates]
        correct += int(point_in_bbox(mvp_official(points, pseudo), context["metadata"][row_id]["target_bbox"]))
    return correct / len(row_ids)


def cala_sequence(context, dev_ids, length=16):
    correct_by_action = {action: correctness(context, action, dev_ids) for action in SHARED_ACTIONS}
    individual = {action: float(np.mean(values)) for action, values in correct_by_action.items()}
    covered = np.zeros(len(dev_ids), dtype=np.bool_)
    selected = []
    records = []
    action_index = {action: index for index, action in enumerate(SHARED_ACTIONS)}
    while len(selected) < length:
        choices = []
        for action in SHARED_ACTIONS:
            if action in selected:
                continue
            resulting = [*selected, action]
            new_covered = covered | correct_by_action[action]
            coverage = float(np.mean(new_covered))
            b3 = b3_accuracy(context, dev_ids, resulting)
            kappa = mean_failure_kappa(correct_by_action, resulting)
            key = (coverage, b3, -kappa, individual[action], -action_index[action])
            choices.append((key, action, coverage, b3, kappa))
        _, action, coverage, b3, kappa = max(choices, key=lambda value: value[0])
        selected.append(action)
        covered |= correct_by_action[action]
        records.append({
            "step": len(selected),
            "action": action_name(action),
            "development_pass_at_n": coverage,
            "development_B3": b3,
            "development_mean_failure_kappa": kappa,
            "individual_development_accuracy": individual[action],
        })
    return tuple(selected), records


def quality_sequence(context, dev_ids):
    values = []
    for index, action in enumerate(SHARED_ACTIONS):
        values.append((float(np.mean(correctness(context, action, dev_ids))), -index, action))
    return tuple(item[2] for item in sorted(values, reverse=True))


def random_sequence(fold):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, fold]))
    order = rng.permutation(len(SHARED_ACTIONS))
    return tuple(SHARED_ACTIONS[index] for index in order)


def evaluate_fold(context, dev_ids, test_ids, actions):
    dev_rows = build_rows(context, dev_ids, actions)
    test_rows = build_rows(context, test_ids, actions)
    tables, priors = fit_ccm(dev_rows)
    outputs = {"B3_mvp": {}, "M1_ccm": {}, "pass_at_n": {}}
    for row in test_rows:
        candidates = row["candidates"]
        points = [candidate["point"] for candidate in candidates]
        pseudo = [{"coverage": candidate.get("coverage", 0), "region": candidate["region"]} for candidate in candidates]
        b3 = mvp_official(points, pseudo)
        m1 = candidates[ccm_select(row, tables, priors)]["point"]
        outputs["B3_mvp"][row["id"]] = point_in_bbox(b3, row["target_bbox"])
        outputs["M1_ccm"][row["id"]] = point_in_bbox(m1, row["target_bbox"])
        outputs["pass_at_n"][row["id"]] = any(point_in_bbox(point, row["target_bbox"]) for point in points)
    return outputs


def merge_outputs(target, source):
    for method, values in source.items():
        overlap = set(target[method]) & set(values)
        if overlap:
            raise ValueError(f"CALA duplicate held-out outputs: {method}/{next(iter(overlap))}")
        target[method].update(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = load_bank()
    methods = ("V_only", "Uniform_Mixed", "Quality_Only", "Random", "CALA_S")
    outputs = {method: {budget: {rule: {} for rule in ("B3_mvp", "M1_ccm", "pass_at_n")} for budget in BUDGETS} for method in methods}
    fold_sequences = {}
    for fold in range(5):
        dev_ids, test_ids = split_ids(context, fold)
        cala, records = cala_sequence(context, dev_ids)
        sequences = {
            "V_only": V_ONLY_SEQUENCE,
            "Uniform_Mixed": UNIFORM_SEQUENCE,
            "Quality_Only": quality_sequence(context, dev_ids),
            "Random": random_sequence(fold),
            "CALA_S": cala,
        }
        fold_sequences[str(fold)] = {
            "development_rows": len(dev_ids),
            "test_rows": len(test_ids),
            "sequences": {method: [action_name(action) for action in sequence[:16]] for method, sequence in sequences.items()},
            "CALA_S_steps": records,
        }
        for method, sequence in sequences.items():
            for budget in BUDGETS:
                selected = sequence[:budget]
                if len(selected) != budget or len(set(selected)) != budget:
                    raise ValueError(f"CALA budget mismatch: {method}/fold{fold}/N{budget}")
                merge_outputs(outputs[method][budget], evaluate_fold(context, dev_ids, test_ids, selected))
    rows = [context["metadata"][row_id] for row_id in context["row_ids"]]
    accuracy = {
        method: {
            str(budget): {rule: sum(values.values()) / len(values) for rule, values in outputs[method][budget].items()}
            for budget in BUDGETS
        }
        for method in methods
    }
    comparisons = {}
    for method in ("CALA_S", "Quality_Only", "V_only", "Random"):
        for budget in BUDGETS:
            for rule in ("B3_mvp", "M1_ccm", "pass_at_n"):
                record = paired_bootstrap(
                    rows,
                    outputs[method][budget][rule],
                    outputs["Uniform_Mixed"][budget][rule],
                    resamples=10000,
                    seed=SEED,
                )
                record.update({
                    "left": f"{method}/N{budget}/{rule}",
                    "right": f"Uniform_Mixed/N{budget}/{rule}",
                    "left_accuracy": accuracy[method][str(budget)][rule],
                    "right_accuracy": accuracy["Uniform_Mixed"][str(budget)][rule],
                })
                comparisons[f"{method}_N{budget}_{rule}_vs_Uniform"] = record
    primary = comparisons["CALA_S_N12_B3_mvp_vs_Uniform"]
    pass_primary = comparisons["CALA_S_N12_pass_at_n_vs_Uniform"]
    result = {
        "schema_version": 1,
        "status": "PASS",
        "rows": 1581,
        "fold_rows": context["fold_rows"],
        "budgets": list(BUDGETS),
        "accuracy": accuracy,
        "comparisons": comparisons,
        "fold_sequences": fold_sequences,
        "sources": {
            "protocol_sha256": sha256_file(RUN_DIR / "configs/protocol.yaml"),
            "L1_RESULTS_sha256": sha256_file(ROOT / "runs/allocation-law/2026-08-01/L1_RESULTS.json"),
            "L2_RESULTS_sha256": sha256_file(ROOT / "runs/allocation-law/2026-08-01/L2_RESULTS.json"),
            "X3_RESULTS_sha256": sha256_file(ROOT / "runs/diversity-axis/2026-08-02/x3_curve_stats.json"),
        },
        "primary": {
            "comparison": "CALA_S_N12_B3_mvp_vs_Uniform",
            "MDE": MDE,
            "delta_positive": primary["point_delta"] > 0,
            "ci_99_lower_positive": primary["ci_99"][0] > 0,
            "delta_above_MDE": primary["point_delta"] > MDE,
            "pass_at_12_non_decreasing": pass_primary["point_delta"] >= 0,
        },
    }
    result["primary"]["success"] = all(value for key, value in result["primary"].items() if key not in ("comparison", "MDE"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accuracy": accuracy, "primary": result["primary"], "primary_comparison": primary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
