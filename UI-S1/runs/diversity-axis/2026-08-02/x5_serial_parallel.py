import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    l1 = json.loads((ROOT / "runs/allocation-law/2026-08-01/L1_RESULTS.json").read_text())
    result = {
        "schema_version": 1,
        "status": "BLOCKED_INCOMPLETE_TOPOLOGY_TRIANGLE",
        "budget": 12,
        "available": {
            "pure_parallel": {
                "pool": "three_lineages_x_four_fixed_views",
                "accuracy": l1["evaluations"]["mixed"]["12"]["accuracy"],
            }
        },
        "unavailable": {
            "pure_serial": "requires frozen adaptive 12-step lineage trace",
            "hybrid": "X2 Q4 has one conditional fourth step per lineage, not four serially conditioned steps",
        },
        "related_X2_Q4": {
            "status": "AVAILABLE_BUT_NOT_X5_HYBRID_ESTIMAND",
            "reason": "each K=3 microchain has three parallel global samples and only one conditional branch",
        },
        "triangle_comparison": "NOT_EVALUATED",
        "best_region": "NOT_EVALUATED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()