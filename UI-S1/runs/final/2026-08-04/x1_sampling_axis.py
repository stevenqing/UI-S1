import argparse
import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCE_PATH = ROOT / "runs/scaleup/2026-08-02/z5_sampling_axis.json"
EXPECTED_SOURCE_SHA256 = "e315041a2a15ef544fda72c7578cc92ea4ef0827cf048a774eb479943973483c"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_path):
    source_hash = sha256_file(SOURCE_PATH)
    source = json.loads(SOURCE_PATH.read_text())
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"X1 source hash mismatch: {source_hash}")
    if source["source"] != {
        "model": "GTA1-7B",
        "samples_per_row": 16,
        "temperature": 0.5,
        "top_p": 0.95,
    }:
        raise ValueError("X1 source protocol mismatch")
    if source["budgets"] != [4, 8, 12, 16] or source["rows"] != 1581:
        raise ValueError("X1 coverage mismatch")
    for pool in source["pools"].values():
        for budget in source["budgets"]:
            record = pool[str(budget)]
            if record["valid_candidates"]["complete_rows"] != 1581:
                raise ValueError(f"X1 incomplete candidates at N={budget}")

    primary = source["slopes"]["S_only"]["GUI_RC"]
    b3 = source["slopes"]["S_only"]["B3_mvp"]
    significantly_negative = primary["point_slope_per_forward"] < 0 and primary["ci_99"][1] < 0
    title_scope = "single_model_diversity_axis" if significantly_negative else "fixed_view_allocation_axis"
    if title_scope != source["prediction"]["title_scope"]:
        raise ValueError("X1 title-scope adjudication mismatch")

    result = {
        "schema_version": 1,
        "status": "PASS_REUSED_COMPLETE_INFERENCE",
        "new_gpu_inference_launched": False,
        "rows": source["rows"],
        "source": source["source"],
        "budgets": source["budgets"],
        "curves": source["pools"],
        "slopes": {
            "S_only_GUI_RC": primary,
            "S_only_B3": b3,
            "sampling_plus_views_GUI_RC": source["slopes"]["sampling_plus_views"]["GUI_RC"],
            "sampling_plus_views_B3": source["slopes"]["sampling_plus_views"]["B3_mvp"],
        },
        "adjudication": {
            "S_only_GUI_RC_point_negative": primary["point_slope_per_forward"] < 0,
            "S_only_GUI_RC_ci_strictly_negative": primary["ci_99"][1] < 0,
            "S_only_B3_ci_strictly_negative": b3["ci_99"][1] < 0,
            "title_scope": title_scope,
            "interpretation": (
                "The S-only GUI-RC point slope is negative but its 99% interval crosses zero; "
                "the B3 slope is near zero. Sampling does not support a general single-model "
                "diversity-axis claim."
            ),
        },
        "provenance": {
            "source_path": str(SOURCE_PATH.relative_to(ROOT)),
            "source_sha256": source_hash,
            "underlying_trace": source["provenance"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_DIR / "x1_sampling_axis.json")
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()