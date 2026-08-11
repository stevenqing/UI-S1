import argparse
import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
UTILITY_DIR = ROOT / "runs/lsa-utility/2026-08-11"
sys.path.insert(0, str(UTILITY_DIR))

from behavior_policy import apply_policy, fit_final_policies
from utility_common import load_banks, load_cev
from vus_data import sha256_file


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=RUN_DIR / "data/public_records.jsonl")
    parser.add_argument("--output", type=Path, default=RUN_DIR / "data/fallbacks.jsonl")
    args = parser.parse_args()
    records = load_jsonl(args.records)
    banks = load_banks()
    cev = load_cev()
    policies = {fold: fit_final_policies(banks, fold, cev) for fold in range(5)}
    output = []
    for record in records:
        row = banks[record["arm"]][record["benchmark"]][record["row_id"]]
        if row.fold != record["fold"]:
            raise ValueError(f"fold mismatch: {record['sample_key']}")
        fallback = apply_policy(row, policies[row.fold][record["benchmark"]][record["arm"]])
        expected = bool(cev["outputs"][record["benchmark"]][record["arm"]]["CEV_A"][record["row_id"]])
        observed = bool(row.candidates[fallback].success)
        if observed != expected:
            raise ValueError(f"V-K2 fallback mismatch: {record['sample_key']}")
        output.append({
            "schema_version": 1,
            "sample_key": record["sample_key"],
            "outer_fold": row.fold,
            "fallback_index": fallback,
        })
    if len(output) != len({row["sample_key"] for row in output}):
        raise ValueError("duplicate fallback keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "records": len(output),
        "mismatches": 0,
        "sha256": sha256_file(args.output),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
