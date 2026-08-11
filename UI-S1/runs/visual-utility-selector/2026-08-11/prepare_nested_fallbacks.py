import argparse
import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
UTILITY_DIR = ROOT / "runs/lsa-utility/2026-08-11"
sys.path.insert(0, str(UTILITY_DIR))

from behavior_policy import apply_policy, fit_final_policies, fit_inner_policies, load_cev_config
from utility_common import load_banks, load_cev
from vus_data import sha256_file


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=RUN_DIR / "data/public_records.jsonl")
    parser.add_argument("--output", type=Path, default=RUN_DIR / "data/nested_fallbacks.jsonl")
    args = parser.parse_args()
    records = load_jsonl(args.records)
    by_fold = {fold: [row for row in records if row["fold"] == fold] for fold in range(5)}
    if any(not values for values in by_fold.values()):
        raise ValueError("empty VUS fold")
    banks = load_banks()
    cev = load_cev()
    cev_config = load_cev_config()
    output = []
    reports = []
    for outer_fold in range(5):
        dev_folds = [fold for fold in range(5) if fold != outer_fold]
        final_policies = fit_final_policies(banks, outer_fold, cev)
        for record in by_fold[outer_fold]:
            row = banks[record["arm"]][record["benchmark"]][record["row_id"]]
            fallback = apply_policy(row, final_policies[record["benchmark"]][record["arm"]])
            expected = bool(cev["outputs"][record["benchmark"]][record["arm"]]["CEV_A"][record["row_id"]])
            if bool(row.candidates[fallback].success) != expected:
                raise ValueError(f"V-K2 outer fallback mismatch: {record['sample_key']}")
            output.append({
                "schema_version": 1,
                "context_key": f"outer{outer_fold}/{record['sample_key']}",
                "outer_fold": outer_fold,
                "role": "test",
                "sample_key": record["sample_key"],
                "fallback_index": fallback,
            })
        for holdout_fold in dev_folds:
            train_folds = [fold for fold in dev_folds if fold != holdout_fold]
            policies, policy_report = fit_inner_policies(banks, train_folds, holdout_fold, cev_config)
            reports.append({
                "outer_fold": outer_fold,
                "holdout_fold": holdout_fold,
                "train_folds": train_folds,
                "behavior_policy": policy_report,
            })
            for record in by_fold[holdout_fold]:
                row = banks[record["arm"]][record["benchmark"]][record["row_id"]]
                fallback = apply_policy(row, policies[record["benchmark"]][record["arm"]])
                output.append({
                    "schema_version": 1,
                    "context_key": f"outer{outer_fold}/{record['sample_key']}",
                    "outer_fold": outer_fold,
                    "role": "dev",
                    "sample_key": record["sample_key"],
                    "fallback_index": fallback,
                })
    if len(output) != 5 * len(records):
        raise ValueError(f"nested fallback count mismatch: {len(output)}")
    keys = [row["context_key"] for row in output]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate nested fallback context keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in sorted(output, key=lambda row: row["context_key"]):
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "PASS_EXACT_NESTED_FALLBACKS",
        "public_records": len(records),
        "contexts": len(output),
        "outer_test_mismatches": 0,
        "inner_policy_fits": len(reports),
        "sha256": sha256_file(args.output),
        "inner_reports": reports,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "public_records", "contexts", "outer_test_mismatches", "inner_policy_fits", "sha256")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()