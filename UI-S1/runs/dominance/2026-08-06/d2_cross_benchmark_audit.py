import hashlib
import json
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
COMPLEMENTARITY_DIR = ROOT / "runs/complementarity/2026-07-30"
FINAL_DIR = ROOT / "runs/final/2026-08-04"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lane_accuracy(lanes, setting, model):
    lane = lanes[f"{setting}/{model}"]
    return lane["successes"] / lane["rows"]


def pool_anchor(lanes, setting, models):
    members = {model: lane_accuracy(lanes, setting, model) for model in models}
    ordered = sorted(members.values(), reverse=True)
    return {
        "models": models,
        "member_accuracy_unfiltered_manifest": members,
        "mean_member_accuracy": sum(ordered) / len(ordered),
        "strongest_member_accuracy": ordered[0],
        "second_member_accuracy": ordered[1],
        "dominance_gap": ordered[0] - ordered[1],
        "pass_at_n": None,
        "stage_ab": None,
        "weighted_full": None,
        "mean_pairwise_failure_kappa": None,
        "status": "BLOCKED_MISSING_ROW_LEVEL_TRACES",
    }


def main():
    manifest_path = COMPLEMENTARITY_DIR / "rows_manifest.json"
    config_path = FINAL_DIR / "configs/t1_t2_pools.yaml"
    runner_path = FINAL_DIR / "t1_t2_transfer.py"
    rows_path = COMPLEMENTARITY_DIR / "rows.parquet"
    manifest = json.loads(manifest_path.read_text())
    config = yaml.safe_load(config_path.read_text())

    lane_roots = {
        "AndroidControl": ROOT / "runs/androidcontrol-rft/2026-07-29/artifacts",
        "Mind2Web": ROOT / "runs/mind2web-tongui/2026-07-28/artifacts",
    }
    present_prediction_files = {
        name: len(list(path.glob("**/predictions.jsonl"))) if path.is_dir() else 0
        for name, path in lane_roots.items()
    }
    rows_present = rows_path.is_file()
    if rows_present:
        actual_rows_hash = sha256_file(rows_path)
        if actual_rows_hash != manifest["rows_parquet_sha256"]:
            raise ValueError("D2 rows.parquet hash mismatch")
    else:
        actual_rows_hash = None

    anchors = {}
    sections = (
        ("T1_mind2web", "mind2web", "visual"),
        ("T2_androidcontrol", "androidcontrol", "low"),
    )
    for section, manifest_key, setting in sections:
        lanes = manifest[manifest_key]["lanes"]
        anchors[section] = {
            pool_name: pool_anchor(lanes, setting, pool["models"])
            for pool_name, pool in config[section]["pools"].items()
        }

    ready = rows_present
    result = {
        "schema_version": 1,
        "status": "READY" if ready else "BLOCKED_MISSING_ROW_LEVEL_TRACES",
        "execution": "NOT_RUN" if not ready else "READY_FOR_FROZEN_RUNNER",
        "rows_parquet": {
            "path": str(rows_path.relative_to(ROOT)),
            "present": rows_present,
            "expected_sha256": manifest["rows_parquet_sha256"],
            "actual_sha256": actual_rows_hash,
            "expected_rows": manifest["rows"],
        },
        "prediction_file_counts": present_prediction_files,
        "pool_table": anchors,
        "mind2web_control": {
            "strongest_member_fixed": config["T1_mind2web"]["fixed_fact"]["strongest_member_both_pools"],
            "cross_minus_same_mean_quality_pp": config["T1_mind2web"]["fixed_fact"]["manifest_step_micro_cross_minus_same_mean_member_quality_pp"],
            "mixed_metrics_available": ready,
        },
        "androidcontrol_confound": config["T2_androidcontrol"]["confound"],
        "quality_matched_control": (
            "AVAILABLE" if config["T2_androidcontrol"]["confound"]["exact_quality_matched_same_family_pool_available"]
            else "UNAVAILABLE_DECLARED_CONFOUND"
        ),
        "D_K3": "NOT_ADJUDICATED_MISSING_MIND2WEB_TRANSFER_OUTPUT",
        "paper_action": "NO_CROSS_BENCHMARK_TRANSFER_CLAIM; keep allocation claims ScreenSpot-Pro-specific",
        "rebuild_attempt": {
            "status": "FAILED_MISSING_SOURCE_PREDICTIONS" if not ready else "NOT_NEEDED",
            "first_missing_path": "runs/androidcontrol-rft/2026-07-29/artifacts/ui-agile-3b/low/predictions.jsonl" if not ready else None,
            "note": "score.json and audit.json do not contain row-level joint correctness or coordinates and cannot replace predictions.jsonl",
        },
        "sources": {
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256_file(manifest_path)},
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256_file(config_path)},
            "runner": {"path": str(runner_path.relative_to(ROOT)), "sha256": sha256_file(runner_path)},
        },
    }
    output = RUN_DIR / "d2_cross_benchmark_status.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "D_K3": result["D_K3"],
        "paper_action": result["paper_action"],
        "output": str(output.relative_to(ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()