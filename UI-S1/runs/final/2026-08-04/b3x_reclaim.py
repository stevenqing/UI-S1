import argparse
import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCEBIAS = ROOT / "runs/sourcebias/2026-08-03"
CALA = ROOT / "runs/cala/2026-08-03"
NEFF = ROOT / "runs/neff/2026-08-03"
REALLOCATION = ROOT / "runs/reallocation/2026-08-03"
sys.path.insert(0, str(SOURCEBIAS))
sys.path.insert(0, str(CALA))
sys.path.insert(0, str(NEFF))
sys.path.insert(0, str(REALLOCATION))
from b2_lineage_normalized import bootstrap, evaluate, fit_stats
from cala_common import UNIFORM_SEQUENCE, load_bank, split_ids
from cala_static import evaluate_fold
from n4_noa import development_statistics, noa_sequence
from reallocation_common import load_pools, ordered_bins, uncertainty_scores


B2_PATH = SOURCEBIAS / "results/b2_lineage_normalized_24.json"


def selected_variant(b2, fold):
    return b2["reports"]["7B"]["outer_selections"][str(fold)]["selected_variant"]


def rows_for(context, row_ids, actions, fold):
    return [
        {
            "id": row_id,
            "application": context["metadata"][row_id]["application"],
            "target_bbox": context["metadata"][row_id]["target_bbox"],
            "outer_fold": fold,
            "candidates": [context["bank"][action][row_id] for action in actions],
        }
        for row_id in row_ids
    ]


def merge(target, values):
    overlap = set(target) & set(values)
    if overlap:
        raise ValueError(f"B3x duplicate outputs: {len(overlap)}")
    target.update(values)


def evaluate_fold_selected(context, fold, actions, variant, subset_ids=None):
    dev_ids, test_ids = split_ids(context, fold)
    dev_rows = rows_for(context, dev_ids, actions, fold)
    test_rows = rows_for(context, test_ids, actions, fold)
    stats = fit_stats(dev_rows)
    outputs = evaluate(test_rows, variant, stats)
    if subset_ids is not None:
        outputs = {row_id: value for row_id, value in outputs.items() if row_id in subset_ids}
    return outputs


def preflight():
    required = [
        B2_PATH,
        ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl",
        ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18",
        ROOT / "runs/ccm-h2h/2026-07-31/h3/shards/qwen3_views",
        ROOT / "runs/ccm-h2h/2026-07-31/h3/shards/uitars_views",
        ROOT / "runs/allocation-law/2026-08-01/shards",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        return {"status": "BLOCKED_MISSING_ASSETS", "missing": missing}
    b2 = json.loads(B2_PATH.read_text())
    if not b2.get("B2_primary_success"):
        return {"status": "CANCELLED_B2_GATE", "B3x_action": b2.get("B3x_action")}
    return {"status": "READY", "B2": str(B2_PATH.relative_to(ROOT))}


def run(output_path):
    check = preflight()
    if check["status"] != "READY":
        result = {"schema_version": 1, **check, "executed": False}
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    b2 = json.loads(B2_PATH.read_text())
    context = load_bank()
    static = json.loads((CALA / "cala_static_results.json").read_text())
    noa = json.loads((NEFF / "n4_noa.json").read_text())
    reallocation = load_pools()
    rows12 = reallocation["mixed"][12]
    image_sizes = {row_id: row["img_size"] for row_id, row in reallocation["gta1"].items()}
    bins = ordered_bins(rows12, uncertainty_scores(rows12, image_sizes), 5)
    hardest = set(bins[-1])

    selected_outputs = {
        "Uniform_Mixed_N12": {},
        "CALA_S_N12": {},
        "NOA_static_N12": {},
        "R1_hard_N4": {},
        "R1_hard_N24": {},
    }
    for fold in range(5):
        variant = selected_variant(b2, fold)
        dev_ids, test_ids = split_ids(context, fold)
        cala_actions = tuple(
            tuple(name.rsplit("/view", 1))
            for name in static["fold_sequences"][str(fold)]["sequences"]["CALA_S"][:12]
        )
        cala_actions = tuple((model, int(view)) for model, view in cala_actions)
        _, accuracy, correlations = development_statistics(context, dev_ids)
        noa_actions, _ = noa_sequence(accuracy, correlations)
        noa_actions = noa_actions[:12]

        merge(selected_outputs["Uniform_Mixed_N12"], evaluate_fold_selected(context, fold, UNIFORM_SEQUENCE[:12], variant))
        merge(selected_outputs["CALA_S_N12"], evaluate_fold_selected(context, fold, cala_actions, variant))
        merge(selected_outputs["NOA_static_N12"], evaluate_fold_selected(context, fold, noa_actions, variant))
        merge(selected_outputs["R1_hard_N4"], evaluate_fold_selected(context, fold, UNIFORM_SEQUENCE[:4], variant, hardest))
        merge(selected_outputs["R1_hard_N24"], evaluate_fold_selected(context, fold, UNIFORM_SEQUENCE[:24], variant, hardest))

    metadata = [context["metadata"][row_id] for row_id in context["row_ids"]]
    hard_metadata = [row for row in metadata if row["id"] in hardest]
    comparisons = {
        "CALA_S_N12_minus_Uniform_Mixed_N12": bootstrap(metadata, selected_outputs["CALA_S_N12"], selected_outputs["Uniform_Mixed_N12"]),
        "NOA_static_N12_minus_Uniform_Mixed_N12": bootstrap(metadata, selected_outputs["NOA_static_N12"], selected_outputs["Uniform_Mixed_N12"]),
        "R1_hard_selected_N24_minus_N4": bootstrap(hard_metadata, selected_outputs["R1_hard_N24"], selected_outputs["R1_hard_N4"]),
    }
    unified = all(record["point_delta"] >= 0 for record in comparisons.values())
    result = {
        "schema_version": 1,
        "status": "PASS" if unified else "PASS_LIMITED_MECHANISM",
        "executed": True,
        "comparisons": comparisons,
        "unified_mechanism_success": unified,
        "claim_scope": "unified_three_failure_reclaim" if unified else "direct_B2_pool_only",
        "B2_selected_variants": {str(fold): selected_variant(b2, fold) for fold in range(5)},
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_DIR / "b3x_reclaim.json")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
