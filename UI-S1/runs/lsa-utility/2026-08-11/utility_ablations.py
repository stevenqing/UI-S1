import json
from pathlib import Path

import yaml

from utility_train import run


RUN_DIR = Path(__file__).resolve().parent


def compact(result):
    return {
        "feature_mode": result["feature_mode"],
        "fallback_validation": result["fallback_validation"],
        "accuracy": result["accuracy"],
        "folds": [
            {
                "outer_fold": fold["outer_fold"],
                "selected": fold["selected"],
                "test": fold["test"],
            }
            for fold in result["folds"]
        ],
        "outputs": result["outputs"],
    }


def main():
    config = yaml.safe_load((RUN_DIR / "configs/utility_prereg.yaml").read_text())
    if not (RUN_DIR / "utility_main.json").exists():
        raise FileNotFoundError("utility_main.json must be completed before ablations")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "no_MVP_structure": compact(run(config, feature_mode="no_mvp")),
        "absolute_only": compact(run(config, feature_mode="absolute")),
    }
    (RUN_DIR / "utility_ablations.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "no_MVP_structure": result["no_MVP_structure"]["accuracy"],
        "absolute_only": result["absolute_only"]["accuracy"],
        "selections": {
            key: [fold["selected"] for fold in result[key]["folds"]]
            for key in ("no_MVP_structure", "absolute_only")
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()