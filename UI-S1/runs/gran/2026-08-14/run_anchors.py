import importlib.util
import json
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
E1_DIR = ROOT / "runs/close/2026-08-08"
OUTPUT_PATH = RUN_DIR / "ANCHORS.json"
EXPECTED = {
    "mind2web_C_uni_sequential": 0.2668269230769231,
    "screenspot_C_uni_B3": 0.6369386464263125,
    "screenspot_C_uni_A2": 0.6388361796331435,
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    click = json.loads((RUN_DIR / "CLICK_SCOPE.json").read_text())
    if (
        click.get("status") != "CLICK_SCOPE_LOCKED_BEFORE_P_HAT_OR_MARGIN_ACCESS"
        or click.get("selected_strata") != 4
        or click.get("tau_sweep_started") is not False
    ):
        raise PermissionError("GRAN CLICK scope is not locked")
    e1 = load_module(E1_DIR / "e1_arm_aggregator_matrix.py", "gran_e1_anchor")
    config = yaml.safe_load((E1_DIR / "configs/aggregator_map.yaml").read_text())
    mind = e1.mind2web_matrix(config)
    screen = e1.screenspot_matrix(config)
    observed = {
        "mind2web_C_uni_sequential": mind["accuracy"]["C_uni"]["ours"],
        "screenspot_C_uni_B3": screen["accuracy"]["C_uni"]["ours"],
        "screenspot_C_uni_A2": screen["accuracy"]["C_uni"]["A2"],
    }
    differences = {name: observed[name] - expected for name, expected in EXPECTED.items()}
    if any(value != 0.0 for value in differences.values()):
        raise ValueError(f"GRAN implementation anchor mismatch: {differences}")
    result = {
        "schema_version": 1,
        "status": "PASS_GRAN_IMPLEMENTATION_ANCHORS",
        "zero_gpu": True,
        "tau_sweep_started": False,
        "p_hat_computed": False,
        "margin_stratification_started": False,
        "expected": EXPECTED,
        "observed": observed,
        "differences": differences,
        "screenspot_A2_anchor_use": "aggregate_digitwise_implementation_anchor_only",
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()