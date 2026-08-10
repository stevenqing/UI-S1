import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
AGGMATCH = ROOT / "runs/aggmatch/2026-08-09"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(AGGMATCH))

import cev_main as cm
from aggmatch_common import paired_bootstrap
from cev import Candidate, select


def paired(metadata, left, right, config, benchmark, seed_offset):
    differences = {row_id: int(left[row_id]) - int(right[row_id]) for row_id in left}
    return paired_bootstrap(
        metadata,
        differences,
        config["statistics"]["paired_bootstrap_resamples"],
        config["benchmarks"][benchmark]["bootstrap_seed"] + seed_offset,
    )


def difference_in_differences(metadata, left_a, left_b, right_a, right_b, config, seed):
    differences = {
        row_id: (int(left_a[row_id]) - int(left_b[row_id])) - (int(right_a[row_id]) - int(right_b[row_id]))
        for row_id in metadata
    }
    return paired_bootstrap(metadata, differences, config["statistics"]["paired_bootstrap_resamples"], seed)


def distribution(values):
    counts = Counter(values)
    total = sum(counts.values())
    shares = {key: value / total for key, value in sorted(counts.items())}
    entropy = -sum(share * math.log(share) for share in shares.values() if share > 0)
    return {"counts": dict(sorted(counts.items())), "shares": shares, "max_share": max(shares.values()), "entropy_nats": entropy}


def screen_source_and_margin(e1, config, baseline_screen):
    context_module = e1.load_module(e1.CONSOLIDATE / "common.py", "cev_diag_screen_common")
    context = context_module.load_context()
    row_ids = context["row_ids"]
    targets = {row_id: context["metadata"][row_id]["target_bbox"] for row_id in row_ids}
    fold_for_group = context["fold_for_group"]
    fold_for_id = {row_id: fold_for_group[context["metadata"][row_id]["application"]] for row_id in row_ids}
    actions = [(model, view) for view in range(4) for model in e1.SCREEN_MODELS]
    slots = {row_id: [(f"{model}_view{view}", dict(context["bank"][(model, view)][row_id])) for model, view in actions] for row_id in row_ids}
    threshold = config["benchmarks"]["screenspot_pro"]["coordinate_base_tolerance"]
    variants = {"unlimited": None, "cap_1": 1, "cap_2": 2}
    selected_lineages = {name: {} for name in variants}
    successes = {name: {} for name in variants}
    margins = {}
    for outer_fold in range(5):
        outer_dev_ids = [row_id for row_id in row_ids if fold_for_id[row_id] != outer_fold]
        test_ids = [row_id for row_id in row_ids if fold_for_id[row_id] == outer_fold]
        _, reliability = e1.screen_dev_priority(outer_dev_ids, slots, targets)
        for row_id in test_ids:
            candidates = [
                Candidate(action="POINT", coordinate=tuple(candidate["point"]), parameter="", source=slot, reliability=reliability[slot], order=order, payload=candidate, parse_ok=True, lineage=candidate["model"])
                for order, (slot, candidate) in enumerate(slots[row_id])
            ]
            for name, cap in variants.items():
                prediction, details = select(candidates, "G4", threshold, lineage_vote_cap=cap)
                selected_lineages[name][row_id] = prediction.lineage
                successes[name][row_id] = bool(e1.point_in_bbox(prediction.coordinate, targets[row_id]))
                if name == "unlimited":
                    sizes = sorted((len(members) for members in details["classes"]), reverse=True)
                    margins[row_id] = sizes[0] - (sizes[1] if len(sizes) > 1 else 0)
    metadata = {row_id: {"fold": fold_for_id[row_id], "group": context["metadata"][row_id]["application"]} for row_id in row_ids}
    margin_groups = {}
    for label, predicate in (
        ("tie", lambda value: value == 0),
        ("margin_1", lambda value: value == 1),
        ("margin_2plus", lambda value: value >= 2),
    ):
        ids = [row_id for row_id, margin in margins.items() if predicate(margin)]
        if not ids:
            continue
        subset_meta = {row_id: metadata[row_id] for row_id in ids}
        margin_groups[label] = {
            "rows": len(ids),
            "mean_margin": float(np.mean([margins[row_id] for row_id in ids])),
            "CEV_minus_majority": paired(subset_meta, {row_id: successes["unlimited"][row_id] for row_id in ids}, {row_id: baseline_screen["outputs"]["C_uni"]["majority"][row_id] for row_id in ids}, config, "screenspot_pro", 20 + len(margin_groups)),
        }
    return {
        "variants": {
            name: {
                "accuracy": float(np.mean(list(successes[name].values()))),
                "selected_lineage_distribution": distribution(selected_lineages[name].values()),
            }
            for name in variants
        },
        "support_margin": margin_groups,
    }


def mind_margin(e1, config, main, baseline_mind):
    rows = [json.loads(line) for line in (e1.XFER / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    rows_by_id = {row["id"]: row for row in rows}
    image_sizes = {row["id"]: Image.open(ROOT / row["image"]).size for row in rows}
    full = {model: e1.load_unique(e1.XFER / "raw/stage1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    view1 = {model: e1.load_unique(e1.XFER / "raw/stage1/view1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    stage2 = {model: e1.load_unique(e1.XFER / "raw/stage2" / directory) for model, directory in e1.MODEL_DIRS.items()}
    slots = e1.mind_slots(rows_by_id, full, view1, stage2, "C_uni")
    fold_map = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    fold_for_id = {row_id: fold_map[row["website"]] for row_id, row in rows_by_id.items()}
    margins = {}
    for outer_fold in range(5):
        outer_dev_ids = [row_id for row_id in rows_by_id if fold_for_id[row_id] != outer_fold]
        test_ids = [row_id for row_id in rows_by_id if fold_for_id[row_id] == outer_fold]
        priority, _ = e1.dev_mind_statistics(outer_dev_ids, slots, rows_by_id, image_sizes)
        reliability = {slot: float(np.mean([e1.score_prediction(rows_by_id[row_id], prediction, image_sizes[row_id]) for row_id in outer_dev_ids for candidate_slot, _, prediction in slots[row_id] if candidate_slot == slot])) for slot in priority}
        scale = cm.mind_scale(outer_dev_ids, rows_by_id, image_sizes)
        for row_id in test_ids:
            candidates = cm.mind_candidates(e1, slots, reliability, row_id)
            _, details = select(candidates, "G0", scale)
            sizes = sorted((len(members) for members in details["classes"]), reverse=True)
            margins[row_id] = sizes[0] - (sizes[1] if len(sizes) > 1 else 0)
    metadata = main["mind2web"]["metadata"]
    outputs = main["outputs"]["mind2web"]["C_uni"]["CEV_A"]
    sequential = baseline_mind["outputs"]["C_uni"]["ours"]
    groups = {}
    for label, predicate in (
        ("tie", lambda value: value == 0),
        ("margin_1", lambda value: value == 1),
        ("margin_2plus", lambda value: value >= 2),
    ):
        ids = [row_id for row_id, margin in margins.items() if predicate(margin)]
        if not ids:
            continue
        subset_meta = {row_id: metadata[row_id] for row_id in ids}
        groups[label] = {
            "rows": len(ids),
            "mean_margin": float(np.mean([margins[row_id] for row_id in ids])),
            "CEV_minus_sequential": paired(subset_meta, {row_id: outputs[row_id] for row_id in ids}, {row_id: sequential[row_id] for row_id in ids}, config, "mind2web", 30 + len(groups)),
        }
    return groups


def main():
    config = yaml.safe_load((RUN_DIR / "configs/cev_prereg.yaml").read_text())
    main_result = json.loads((RUN_DIR / "cev_main.json").read_text())
    ablations = json.loads((RUN_DIR / "cev_ablations.json").read_text())
    e1 = cm.load_module(cm.CLOSE / "e1_arm_aggregator_matrix.py", "cev_diag_e1")
    baseline_config = yaml.safe_load((cm.CLOSE / "configs/aggregator_map.yaml").read_text())
    baseline_mind = e1.mind2web_matrix(baseline_config)
    baseline_screen = e1.screenspot_matrix(baseline_config)
    mind_outputs = main_result["outputs"]["mind2web"]
    mind_meta = main_result["mind2web"]["metadata"]

    source_margin = screen_source_and_margin(e1, config, baseline_screen)
    inherited_b2 = json.loads((ROOT / "runs/sourcebias/2026-08-03/results/b2_lineage_normalized.json").read_text())["reports"]["72B"]
    p_a = {
        "status": "CONTAMINATED_INHERITED_DIAGNOSTIC",
        "screenspot_7B": source_margin["variants"],
        "screenspot_cap_1_minus_unlimited_pp": 100 * (source_margin["variants"]["cap_1"]["accuracy"] - source_margin["variants"]["unlimited"]["accuracy"]),
        "screenspot_cap_2_minus_unlimited_pp": 100 * (source_margin["variants"]["cap_2"]["accuracy"] - source_margin["variants"]["unlimited"]["accuracy"]),
        "inherited_72B_lineage_normalization": {
            "nested_LN": inherited_b2["accuracy"]["nested_LN"],
            "B3": inherited_b2["accuracy"]["B3_mvp"],
            "best_single": inherited_b2["accuracy"]["best_single_matched_bank_view0"],
            "LN_minus_B3": inherited_b2["comparisons"]["vs_B3"],
            "LN_minus_best_single": inherited_b2["comparisons"]["vs_best_single_matched_bank_view0"],
        },
        "interpretation": "vote caps trade source concentration against density signal; cap 1 harms 7B while historical 72B normalization improves B3 but approaches best-single",
    }

    p_b = {
        "status": "CONTAMINATED_ANCHOR",
        "supported": main_result["screenspot_pro"]["accuracy"]["C_uni"]["CEV_A"] == main_result["baseline_accuracy"]["screenspot_pro"]["C_uni"]["A2"],
        "selected_granularity_all_folds": "G4",
    }
    global_granularities = [fold["arms"]["C_uni"]["global_configuration"]["granularity"] for fold in main_result["mind2web"]["folds"]]
    p_c = {
        "status": "PROSPECTIVE_MOTIVATED_BY_F1",
        "supported": Counter(global_granularities)["G0"] >= 4,
        "global_granularity_counts": dict(Counter(global_granularities)),
        "CEV_A_accuracy": main_result["mind2web"]["accuracy"]["C_uni"]["CEV_A"],
        "majority_accuracy": main_result["baseline_accuracy"]["mind2web"]["C_uni"]["majority"],
    }
    action_selections = {}
    for fold in main_result["mind2web"]["folds"]:
        for action, values in fold["arms"]["C_uni"]["action_configurations"].items():
            action_selections.setdefault(action, []).append({"outer_fold": fold["outer_fold"], **values})
    p_d = {
        "status": "PROSPECTIVE",
        "supported": all(value["granularity"] == "G0" for value in action_selections.get("CLICK", [])),
        "action_selections": action_selections,
        "limitation": "TYPE/SELECT predicted plurality rows are below 30 in every fold and therefore back off; no independent sparse-action conclusion",
    }

    ceva_arm = paired(mind_meta, mind_outputs["C_cond"]["CEV_A"], mind_outputs["C_uni"]["CEV_A"], config, "mind2web", 40)
    sequential_arm = paired(mind_meta, baseline_mind["outputs"]["C_cond"]["ours"], baseline_mind["outputs"]["C_uni"]["ours"], config, "mind2web", 41)
    absorption = difference_in_differences(
        mind_meta,
        mind_outputs["C_cond"]["CEV_A"], mind_outputs["C_uni"]["CEV_A"],
        baseline_mind["outputs"]["C_cond"]["ours"], baseline_mind["outputs"]["C_uni"]["ours"],
        config, config["benchmarks"]["mind2web"]["bootstrap_seed"] + 42,
    )
    p_e = {
        "status": "PROSPECTIVE",
        "supported": abs(ceva_arm["point_delta"]) < abs(sequential_arm["point_delta"]),
        "C_cond_minus_C_uni_CEV_A": ceva_arm,
        "C_cond_minus_C_uni_sequential": sequential_arm,
        "difference_in_differences": absorption,
    }
    p_f = {
        "status": "PROSPECTIVE",
        "supported": Counter(global_granularities).most_common(1)[0][1] >= 4,
        "mind2web_global_granularities": global_granularities,
        "screenspot_global_granularities": ["G4"] * 5,
        "C_K5": main_result["gates"]["C_K5"],
        "tolerance_rank_flip": main_result["gates"]["C_K5_diagnostics"]["central_tolerance_rank_flip"],
        "interpretation": "selected endpoint is stable, but unused coordinate-tolerance rankings flip; universal tolerance-free wording is prohibited",
    }

    action_strata = {}
    for index, action in enumerate(("CLICK", "TYPE", "SELECT")):
        ids = [row_id for row_id, row in mind_meta.items() if row["action"] == action]
        subset_meta = {row_id: mind_meta[row_id] for row_id in ids}
        action_strata[action] = paired(
            subset_meta,
            {row_id: mind_outputs["C_uni"]["CEV_A"][row_id] for row_id in ids},
            {row_id: baseline_mind["outputs"]["C_uni"]["majority"][row_id] for row_id in ids},
            config, "mind2web", 50 + index,
        )
    screen_margins = source_margin["support_margin"]
    mind_margins = mind_margin(e1, config, main_result, baseline_mind)
    screen_values = [value["CEV_minus_majority"]["point_delta"] for value in screen_margins.values()]
    p_g = {
        "status": "PROSPECTIVE_MECHANISM_DIAGNOSTIC",
        "supported": len(screen_values) >= 2 and max(screen_values) > min(screen_values),
        "screenspot_coordinate_support_margin": screen_margins,
        "mind2web_action_support_margin": mind_margins,
        "action_strata_CEV_A_minus_majority": action_strata,
        "interpretation": "support margin is reported directly; action-space dimensionality alone is not treated as causal",
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "P_A": p_a,
        "P_B": p_b,
        "P_C": p_c,
        "P_D": p_d,
        "P_E": p_e,
        "P_F": p_f,
        "P_G": p_g,
        "predictions_supported": {key: value.get("supported") for key, value in (("P_A", p_a), ("P_B", p_b), ("P_C", p_c), ("P_D", p_d), ("P_E", p_e), ("P_F", p_f), ("P_G", p_g))},
        "ablation_reference": "cev_ablations.json",
    }
    (RUN_DIR / "cev_predictions.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions_supported": result["predictions_supported"], "P_E": p_e, "P_F": p_f, "P_G_summary": {"supported": p_g["supported"], "screenspot": screen_margins, "mind2web": mind_margins}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()