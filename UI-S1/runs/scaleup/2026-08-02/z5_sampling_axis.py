import argparse
import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCE = ROOT / "runs/closing/2026-08-02/f2_sampling_axis.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    if source["status"] != "PASS" or source["budgets"] != [4, 8, 12, 16]:
        raise ValueError("Z5 source protocol mismatch")
    if source["source"] != {"model": "GTA1-7B", "samples_per_row": 16, "temperature": 0.5, "top_p": 0.95}:
        raise ValueError("Z5 sampling source mismatch")
    primary = source["slopes"]["S_only"]["GUI_RC"]
    result = {
        "schema_version": 1,
        "status": "PASS_REUSED_COMPLETE_CLOSING_INFERENCE",
        "rows": 1581,
        "budgets": source["budgets"],
        "source": source["source"],
        "pools": source["pools"],
        "slopes": source["slopes"],
        "prediction": source["prediction"],
        "adjudication": {
            "point_slope_negative": primary["point_slope_per_forward"] < 0,
            "ci_strictly_negative": primary["ci_99"][1] < 0,
            "title_scope": "single_model_diversity_axis" if primary["ci_99"][1] < 0 else "fixed_view_allocation_axis",
        },
        "provenance": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "duplicate_inference_launched": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["adjudication"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()