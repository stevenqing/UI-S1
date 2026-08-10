import json
from pathlib import Path

import yaml

from lsa_common import feature_names, load_rows
from lsa_train import BENCHMARKS, run_variant


RUN_DIR = Path(__file__).resolve().parent


def indices_for_variant(name, names):
    if name == "reliability_only":
        allowed = {"parse_ok", "coordinate_present", "parameter_present", "source_reliability"}
        return [index for index, feature in enumerate(names) if feature in allowed or feature.startswith("action_")]
    if name == "no_geometry":
        return [index for index, feature in enumerate(names) if "coordinate" not in feature and "lineage_support" not in feature]
    if name == "no_action":
        return [index for index, feature in enumerate(names) if "action" not in feature]
    if name == "no_parameter":
        return [index for index, feature in enumerate(names) if "parameter" not in feature]
    raise ValueError(name)


def compact(result):
    return {
        "benchmarks": result["benchmarks"],
        "feature_indices": result["feature_indices"],
        "accuracy": result["accuracy"],
        "folds": result["folds"],
        "outputs": result["outputs"],
    }


def main():
    config = yaml.safe_load((RUN_DIR / "configs/lsa_prereg.yaml").read_text())
    if config["status"] != "FROZEN_BEFORE_LSA_RESULTS":
        raise ValueError("LSA preregistration is not frozen")
    banks = load_rows()
    names = feature_names()
    variants = {}
    for benchmark in BENCHMARKS:
        variants[f"within_safe_{benchmark}"] = compact(run_variant(config, banks, (benchmark,)))
    for name in ("reliability_only", "no_geometry", "no_action", "no_parameter"):
        variants[name] = compact(run_variant(config, banks, BENCHMARKS, indices_for_variant(name, names)))
    result = {
        "schema_version": 1,
        "status": "PASS",
        "feature_names": names,
        "variants": variants,
    }
    (RUN_DIR / "lsa_variants.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: value["accuracy"] for name, value in variants.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()