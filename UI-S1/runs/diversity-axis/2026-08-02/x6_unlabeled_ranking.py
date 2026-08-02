import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    l2 = json.loads((ROOT / "runs/allocation-law/2026-08-01/L2_RESULTS.json").read_text())
    development = {
        name: {
            "mean_dev_failure_kappa": pool["mean_dev_kappa"],
            "pass_at_12": pool["accuracy"]["pass_at_n"],
            "M1_ccm": pool["accuracy"]["M1_ccm"],
        }
        for name, pool in l2["pools"].items()
    }
    result = {
        "schema_version": 1,
        "status": "BLOCKED_NO_HELDOUT_POOLS",
        "development_pool_count": len(development),
        "development_inventory": development,
        "fit_allowed": True,
        "heldout_sources": {"X2": "UNAVAILABLE", "X5": "UNAVAILABLE"},
        "heldout_spearman": "NOT_EVALUATED",
        "prediction_threshold": 0.7,
        "prediction_X6": "NOT_EVALUATED",
        "warning": "The eight L2 pools cannot serve as both fitting and held-out evaluation pools.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "development_pool_count": len(development)}, indent=2))


if __name__ == "__main__":
    main()