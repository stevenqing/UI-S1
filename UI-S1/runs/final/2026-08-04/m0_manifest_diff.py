import argparse
import hashlib
import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
CLOSING_DIR = ROOT / "runs/closing/2026-08-02"
sys.path.insert(0, str(H1_DIR))
sys.path.insert(0, str(CLOSING_DIR))
from aggregators_coord import official_groups
from closing_common import load_closing_pools
from f1_paired_bootstrap import paired_bootstrap


MODEL_ORDER = (
    "GTA1-7B",
    "Qwen3-VL-8B-Instruct",
    "UI-TARS-7B-SFT",
)
EXPECTED_HISTORICAL = 0.6363061353573688
EXPECTED_CANONICAL = 0.6369386464263125
EXPECTED_M1 = 0.6382036685641999


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def candidate_key(candidate):
    return candidate["model"], int(candidate["view_index"])


def selected_candidate(candidates):
    points = [candidate["point"] for candidate in candidates]
    groups = official_groups(points)
    scored = []
    for group_index, group in enumerate(groups):
        coverage = sum(candidates[index].get("coverage", 0) for index in group) / len(group)
        scored.append((len(group) + coverage / 1000, -group_index, group))
    winning_group = max(scored)[2]
    selected_index = max(
        winning_group,
        key=lambda index: (candidates[index].get("coverage", 0), -index),
    )
    return selected_index, winning_group


def historical_order(candidates):
    model_rank = {model: index for index, model in enumerate(MODEL_ORDER)}
    return sorted(
        candidates,
        key=lambda candidate: (
            model_rank[candidate["model"]],
            int(candidate["view_index"]),
        ),
    )


def describe_selection(candidates):
    selected_index, winning_group = selected_candidate(candidates)
    selected = candidates[selected_index]
    return {
        "selected_index": selected_index,
        "selected_key": list(candidate_key(selected)),
        "selected_point": list(map(float, selected["point"])),
        "selected_coverage": float(selected.get("coverage", 0)),
        "winning_group_indices": list(winning_group),
        "winning_group_keys": [list(candidate_key(candidates[index])) for index in winning_group],
        "winning_group_lineages": [candidates[index]["model"] for index in winning_group],
    }


def required_assets():
    return (
        ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl",
        ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18",
        ROOT / "runs/ccm-h2h/2026-07-31/h3/shards/qwen3_views",
        ROOT / "runs/ccm-h2h/2026-07-31/h3/shards/uitars_views",
        ROOT / "runs/allocation-law/2026-08-01/shards",
    )


def preflight():
    missing = [str(path.relative_to(ROOT)) for path in required_assets() if not path.exists()]
    return {"status": "READY" if not missing else "BLOCKED_MISSING_FROZEN_ASSETS", "missing": missing}


def run(output_path):
    check = preflight()
    if check["status"] != "READY":
        raise FileNotFoundError(json.dumps(check, sort_keys=True))

    _, pools = load_closing_pools()
    mixed = pools["mixed_N12"]
    v_only = pools["v_only_N12"]
    canonical_units = [candidate_key(candidate) for candidate in mixed["rows"][0]["candidates"]]
    expected_canonical = [
        (model, view)
        for view in range(4)
        for model in MODEL_ORDER
    ]
    if canonical_units != expected_canonical:
        raise ValueError(f"canonical Mixed N12 order mismatch: {canonical_units}")

    historical_outputs = {}
    canonical_outputs = {}
    coordinate_flips = []
    correctness_flips = []
    for row in mixed["rows"]:
        canonical_candidates = row["candidates"]
        historical_candidates = historical_order(canonical_candidates)
        if {candidate_key(value) for value in canonical_candidates} != {
            candidate_key(value) for value in historical_candidates
        }:
            raise ValueError(f"candidate set mismatch: {row['id']}")
        historical = describe_selection(historical_candidates)
        canonical = describe_selection(canonical_candidates)
        historical_correct = point_in_bbox(historical["selected_point"], row["target_bbox"])
        canonical_correct = point_in_bbox(canonical["selected_point"], row["target_bbox"])
        historical_outputs[row["id"]] = historical_correct
        canonical_outputs[row["id"]] = canonical_correct
        if historical["selected_point"] != canonical["selected_point"]:
            record = {
                "id": row["id"],
                "application": row["application"],
                "target_bbox": list(map(float, row["target_bbox"])),
                "historical_H3_model_major": {**historical, "correct": historical_correct},
                "canonical_L1_view_major": {**canonical, "correct": canonical_correct},
                "cross_lineage_selection_flip": historical["selected_key"][0] != canonical["selected_key"][0],
            }
            coordinate_flips.append(record)
            if historical_correct != canonical_correct:
                correctness_flips.append(record)

    historical_accuracy = sum(historical_outputs.values()) / len(historical_outputs)
    canonical_accuracy = sum(canonical_outputs.values()) / len(canonical_outputs)
    m1_accuracy = mixed["evaluation"]["accuracy"]["M1_ccm"]
    if historical_accuracy != EXPECTED_HISTORICAL:
        raise ValueError(f"historical B3 mismatch: {historical_accuracy}")
    if canonical_accuracy != EXPECTED_CANONICAL:
        raise ValueError(f"canonical B3 mismatch: {canonical_accuracy}")
    if m1_accuracy != EXPECTED_M1:
        raise ValueError(f"M1 mismatch: {m1_accuracy}")
    rescues = sum(
        not record["historical_H3_model_major"]["correct"]
        and record["canonical_L1_view_major"]["correct"]
        for record in correctness_flips
    )
    regressions = sum(
        record["historical_H3_model_major"]["correct"]
        and not record["canonical_L1_view_major"]["correct"]
        for record in correctness_flips
    )
    if rescues - regressions != 1:
        raise ValueError(
            f"M0 expected net one-row gain, found rescues={rescues}, regressions={regressions}"
        )

    comparison = paired_bootstrap(
        mixed["rows"],
        canonical_outputs,
        v_only["evaluation"]["outputs"]["B3_mvp"],
    )
    comparison.update({
        "left": "mixed_N12/canonical_view_major_B3",
        "right": "v_only_N12/B3",
        "left_accuracy": canonical_accuracy,
        "right_accuracy": v_only["evaluation"]["accuracy"]["B3_mvp"],
    })
    result = {
        "schema_version": 1,
        "status": "PASS",
        "canonical_value": canonical_accuracy,
        "historical_value": historical_accuracy,
        "aggregate_net_difference_rows": rescues - regressions,
        "correctness_flip_rows": len(correctness_flips),
        "rescues": rescues,
        "regressions": regressions,
        "coordinate_selection_flips": len(coordinate_flips),
        "correctness_flips": correctness_flips,
        "all_coordinate_flips": coordinate_flips,
        "drop_in_comparison": comparison,
        "ccm_attribution": {
            "M1_accuracy": m1_accuracy,
            "canonical_B3_accuracy": canonical_accuracy,
            "M1_minus_B3": m1_accuracy - canonical_accuracy,
        },
        "manifest_orders": {
            "historical_H3": "model_major_then_view",
            "canonical_L1_CALA": "view_major_then_model",
        },
        "sources": {
            "L1_RESULTS_sha256": sha256_file(ROOT / "runs/allocation-law/2026-08-01/L1_RESULTS.json"),
            "H3_RESULT_sha256": sha256_file(ROOT / "runs/ccm-h2h/2026-07-31/h3_mixed_pool.json"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def self_test():
    candidates = [
        {"model": "GTA1-7B", "view_index": 0, "point": [0, 0], "coverage": 1},
        {"model": "Qwen3-VL-8B-Instruct", "view_index": 0, "point": [10, 0], "coverage": 2},
        {"model": "UI-TARS-7B-SFT", "view_index": 0, "point": [20, 0], "coverage": 3},
    ]
    left = describe_selection(candidates)
    right = describe_selection([candidates[1], candidates[2], candidates[0]])
    if left["selected_key"] == right["selected_key"]:
        raise AssertionError("synthetic order-sensitivity probe did not flip")
    return {"status": "PASS", "left": left["selected_key"], "right": right["selected_key"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_DIR / "m0_manifest_diff.json")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.preflight:
        result = preflight()
    else:
        result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()