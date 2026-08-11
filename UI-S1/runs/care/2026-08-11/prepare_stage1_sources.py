import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
UTILITY = ROOT / "runs/lsa-utility/2026-08-11"
sys.path.insert(0, str(UTILITY))

from utility_common import BENCHMARKS, load_banks


def main():
    banks = load_banks()
    output = []
    for benchmark in BENCHMARKS:
        for row_id, row in sorted(banks["C_uni"][benchmark].items()):
            output.append({
                "schema_version": 1,
                "benchmark": benchmark,
                "row_id": row_id,
                "fold": row.fold,
                "sources": [candidate.source for candidate in row.candidates[:6]],
            })
    if len(output) != 2080 + 1581 or any(len(row["sources"]) != 6 for row in output):
        raise ValueError("CARE stage1 source metadata coverage mismatch")
    path = RUN_DIR / "data/stage1_sources.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    public_text = path.read_text().lower()
    if any(token in public_text for token in ('"success"', 'target_bbox', 'candidate_success', 'positive_dom')):
        raise ValueError("CARE source metadata leaked evaluator fields")
    manifest = {
        "schema_version": 1,
        "status": "PASS_PUBLIC_SOURCE_METADATA",
        "records": len(output),
        "candidate_sources_per_row": 6,
        "contains_success": False,
    }
    (RUN_DIR / "data/stage1_sources.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
