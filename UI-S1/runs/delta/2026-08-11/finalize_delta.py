import json
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(VUS))

from finalize_set_ranker import paired_samples
from set_ranker_data import keyed
from delta_model import CHANNELS
from delta_train import CONFIG_PATH, validate_config


VARIANTS = ("FULL", "VUS_ONLY", "VUS_GLOBAL", "VUS_LOCAL", "RANDOM_PLACEBO", "FIXED_AVERAGE")
ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
BENCHMARKS = ("mind2web", "screenspot_pro")
MDE = {"mind2web": 0.006106589385659482, "screenspot_pro": 0.007}


def statistically_noninferior(value, margin):
    return value["ci_99"][0] > -margin


def load_outers():
    values = []
    for fold in range(5):
        path = RUN_DIR / f"outer/outer-{fold}.json"
        pretest = RUN_DIR / f"outer/outer-{fold}.pretest.json"
        if not path.is_file() or not pretest.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text())
        seal = json.loads(pretest.read_text())
        if (
            value["status"] != "PASS_DELTA_OUTER_COMPLETE"
            or seal["status"] != "PASS_DELTA_SELECTION_FROZEN"
            or fold in seal["opened_development_folds"]
        ):
            raise ValueError(f"invalid DELTA outer {fold}")
        values.append(value)
    return values


def merge(outers):
    output = {
        variant: {
            benchmark: {arm: {method: {} for method in ("safe", "direct", "fallback")} for arm in ARMS}
            for benchmark in BENCHMARKS
        }
        for variant in VARIANTS
    }
    for outer in outers:
        for variant in VARIANTS:
            for benchmark in BENCHMARKS:
                for arm in ARMS:
                    for method in output[variant][benchmark][arm]:
                        values = outer["outputs"][variant][benchmark][arm][method]
                        if set(output[variant][benchmark][arm][method]) & set(values):
                            raise ValueError("duplicate DELTA row")
                        output[variant][benchmark][arm][method].update(values)
    expected = {"mind2web": 2080, "screenspot_pro": 1581}
    for variant in VARIANTS:
        for benchmark in BENCHMARKS:
            for arm in ARMS:
                if any(len(output[variant][benchmark][arm][method]) != expected[benchmark] for method in output[variant][benchmark][arm]):
                    raise ValueError(f"DELTA coverage mismatch: {variant}/{benchmark}/{arm}")
    return output


def comparisons(public, left, right, seed_offset):
    report = {benchmark: {} for benchmark in BENCHMARKS}
    samples_by_benchmark = {}
    for benchmark in BENCHMARKS:
        arm_samples = []
        points = []
        seed = (20260891 if benchmark == "mind2web" else 20260892) + seed_offset
        for arm_index, arm in enumerate(ARMS):
            value, samples = paired_samples(
                public, benchmark, arm,
                left[benchmark][arm]["safe"], right[benchmark][arm]["safe"],
                10000, seed + arm_index,
            )
            report[benchmark][arm] = value
            arm_samples.append(samples)
            points.append(value["point_delta"])
        equal_arm = np.mean(np.stack(arm_samples), axis=0)
        report[benchmark]["equal_arm"] = {
            "point_delta": float(np.mean(points)),
            "ci_99": [float(np.quantile(equal_arm, 0.005)), float(np.quantile(equal_arm, 0.995))],
        }
        samples_by_benchmark[benchmark] = equal_arm
    balanced = np.mean(np.stack([
        samples_by_benchmark[benchmark] / MDE[benchmark] for benchmark in BENCHMARKS
    ]), axis=0)
    return report, {
        "point": float(np.mean(balanced)),
        "ci_99": [float(np.quantile(balanced, 0.005)), float(np.quantile(balanced, 0.995))],
    }


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    validate_config(config)
    if config["mde"] != MDE:
        raise ValueError("DELTA adjudication MDE mismatch")
    outers = load_outers()
    outputs = merge(outers)
    public = keyed(VUS / "data/public_records.jsonl")
    vus = json.loads((VUS / "set_ranker_adjudication.json").read_text())["outputs"]
    controls = {"VUS_SR": vus, **{variant: outputs[variant] for variant in VARIANTS if variant != "FULL"}}
    comparison = {}
    balanced = {}
    for index, (name, control) in enumerate(controls.items()):
        comparison[name], balanced[name] = comparisons(public, outputs["FULL"], control, 1000 + index * 100)

    mind = comparison["VUS_SR"]["mind2web"]["equal_arm"]
    screen_cells = comparison["VUS_SR"]["screenspot_pro"]
    delta1 = mind["ci_99"][0] > 0
    delta2 = all(
        statistically_noninferior(screen_cells[arm], MDE["screenspot_pro"])
        for arm in ARMS
    )
    delta3 = balanced["VUS_SR"]["ci_99"][0] > 0
    delta4 = balanced["VUS_ONLY"]["ci_99"][0] > 0
    delta5 = balanced["RANDOM_PLACEBO"]["ci_99"][0] > 0
    real_indices = [CHANNELS.index(name) for name in CHANNELS[:4]]
    full_gate_by_fold = [outer["mean_gate_mass"]["FULL"] for outer in outers]
    qualified = {
        CHANNELS[index]: sum(fold[index] >= 0.10 for fold in full_gate_by_fold)
        for index in real_indices
    }
    stable_channels = [name for name, count in qualified.items() if count >= 4]
    equivariance = [outer["equivariance_max_error"]["FULL"] for outer in outers]
    delta6 = len(stable_channels) >= 2 and max(equivariance) <= 1e-5
    gates = {
        "DELTA_1_mind2web_ci_positive": delta1,
        "DELTA_2_screenspot_cells_noninferior": delta2,
        "DELTA_3_balanced_ci_positive_vs_VUS": delta3,
        "DELTA_4_balanced_ci_positive_vs_VUS_ONLY": delta4,
        "DELTA_5_balanced_ci_positive_vs_RANDOM_PLACEBO": delta5,
        "DELTA_6_stable_multi_channel_attribution": delta6,
    }
    result = {
        "schema_version": 1,
        "status": "PASS_DELTA_ADJUDICATED",
        "outcome": "DELTA_COMPLEMENTARITY_SUPPORTED" if all(gates.values()) else "DELTA_NOT_SUPPORTED",
        "gates": gates,
        "comparisons": comparison,
        "balanced": balanced,
        "channel_attribution": {
            "channel_order": list(CHANNELS),
            "full_gate_mass_by_fold": full_gate_by_fold,
            "qualified_fold_counts": qualified,
            "stable_channels": stable_channels,
            "equivariance_max_error_by_fold": equivariance,
            "channel_dropout_by_fold": [outer["channel_dropout"] for outer in outers],
        },
        "outer_epochs": [outer["final_epochs"] for outer in outers],
        "outputs": outputs,
    }
    path = RUN_DIR / "delta_adjudication.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "outcome": result["outcome"], "gates": gates,
        "mind2web_vs_VUS": mind,
        "screenspot_vs_VUS": comparison["VUS_SR"]["screenspot_pro"],
        "balanced": balanced,
        "channel_attribution": result["channel_attribution"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
