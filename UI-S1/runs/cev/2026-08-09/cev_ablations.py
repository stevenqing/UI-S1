import json
import sys
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

import cev_main as cm


def compact(result, method):
    return {
        "accuracy": result["accuracy"]["C_uni"][method],
        "folds": [
            {
                "outer_fold": fold["outer_fold"],
                "global_configuration": fold["arms"]["C_uni"]["global_configuration"],
                "selected_variant": fold["arms"]["C_uni"]["selected_variant"],
                "action_configurations": fold["arms"]["C_uni"]["action_configurations"],
            }
            for fold in result["folds"]
        ],
    }


def run_mind(e1, config, options, method):
    result = cm.nested_mind(e1, config, options_override=options, arms=("C_uni",))
    return compact(result, method)


def run_sensitivity(e1, config, cap=None, linkage="complete"):
    cm.LINEAGE_VOTE_CAP = cap
    cm.LINKAGE = linkage
    mind = cm.nested_mind(e1, config, arms=("C_uni",))
    screen = cm.nested_screen(e1, config, arms=("C_uni",))
    return {
        "mind2web": compact(mind, "CEV_A"),
        "screenspot_pro": {
            "accuracy": screen["accuracy"]["C_uni"]["CEV_A"],
            "folds": [
                {
                    "outer_fold": fold["outer_fold"],
                    "configuration": fold["arms"]["C_uni"]["global_configuration"],
                }
                for fold in screen["folds"]
            ],
        },
    }


def main():
    config = yaml.safe_load((RUN_DIR / "configs/cev_prereg.yaml").read_text())
    main_result = json.loads((RUN_DIR / "cev_main.json").read_text())
    if main_result["gates"]["C_K1"]:
        raise ValueError("C-K1 blocks ablations")
    e1 = cm.load_module(cm.CLOSE / "e1_arm_aggregator_matrix.py", "cev_ablation_e1")
    multipliers = config["benchmarks"]["mind2web"]["coordinate_multipliers"]
    thresholds = config["benchmarks"]["mind2web"]["parameter_thresholds"]
    g0 = [{"granularity": "G0"}]
    fixed = [
        {"granularity": "G0"},
        {"granularity": "G1", "coordinate_multiplier": 1.0},
        {"granularity": "G2", "parameter_threshold": 1.0},
        {"granularity": "G3", "coordinate_multiplier": 1.0, "parameter_threshold": 1.0},
    ]
    parameter = [
        {"granularity": "G0"},
        {"granularity": "G1", "coordinate_multiplier": 1.0},
        *({"granularity": "G2", "parameter_threshold": value} for value in thresholds),
        *({"granularity": "G3", "coordinate_multiplier": 1.0, "parameter_threshold": value} for value in thresholds),
    ]

    cm.LINEAGE_VOTE_CAP = None
    cm.LINKAGE = "complete"
    action_endpoint = run_mind(e1, config, g0, "global")
    global_fixed = run_mind(e1, config, fixed, "global")
    action_fixed = run_mind(e1, config, fixed, "action")
    parameter_selected = run_mind(e1, config, parameter, "action")
    full_main = {
        "mind2web": {
            "accuracy": main_result["mind2web"]["accuracy"]["C_uni"]["CEV_A"],
            "folds": [
                {
                    "outer_fold": fold["outer_fold"],
                    "global_configuration": fold["arms"]["C_uni"]["global_configuration"],
                    "selected_variant": fold["arms"]["C_uni"]["selected_variant"],
                    "action_configurations": fold["arms"]["C_uni"]["action_configurations"],
                }
                for fold in main_result["mind2web"]["folds"]
            ],
        },
        "screenspot_pro": {
            "accuracy": main_result["screenspot_pro"]["accuracy"]["C_uni"]["CEV_A"],
        },
    }
    cap_one = run_sensitivity(e1, config, cap=1)
    cap_two = run_sensitivity(e1, config, cap=2)
    single_link = run_sensitivity(e1, config, cap=None, linkage="single")
    cm.LINEAGE_VOTE_CAP = None
    cm.LINKAGE = "complete"

    central_rank_flips = []
    for fold in main_result["mind2web"]["folds"]:
        scores = fold["arms"]["C_uni"]["global_validation_scores"]
        for granularity in ("G1", "G3"):
            central = {key: value for key, value in scores.items() if key.startswith(granularity) and any(f"c{multiplier}" in key for multiplier in (0.75, 1.0, 1.25))}
            if central:
                best = max(central, key=lambda key: (central[key], key))
                central_rank_flips.append({"outer_fold": fold["outer_fold"], "granularity": granularity, "best": best, "scores": central})
    result = {
        "schema_version": 1,
        "status": "PASS",
        "fixed_order": config["ablations"]["fixed_order"],
        "A0_G0_action_endpoint": {"mind2web": action_endpoint},
        "A1_G4_coordinate_endpoint": {"screenspot_pro": {"accuracy": main_result["screenspot_pro"]["accuracy"]["C_uni"]["CEV_A"]}},
        "A2_global_fixed_threshold_granularity": {"mind2web": global_fixed},
        "A3_action_conditional_fixed_threshold": {"mind2web": action_fixed},
        "A4_parameter_threshold_selection": {"mind2web": parameter_selected},
        "A5_full_coordinate_tolerance_selection": full_main,
        "A6_lineage_vote_cap": {"cap_1": cap_one, "cap_2": cap_two, "unlimited": full_main},
        "A7_single_link_sensitivity": single_link,
        "central_tolerance_rank_audit": central_rank_flips,
        "contamination_note": "ScreenSpot cap-1 and linkage outcomes are analysis-only because related EQV cells were disclosed before reconstruction.",
    }
    (RUN_DIR / "cev_ablations.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "A0": action_endpoint["accuracy"],
        "A2": global_fixed["accuracy"],
        "A3": action_fixed["accuracy"],
        "A4": parameter_selected["accuracy"],
        "A5": full_main,
        "cap_1": {key: value["accuracy"] for key, value in cap_one.items()},
        "cap_2": {key: value["accuracy"] for key, value in cap_two.items()},
        "single_link": {key: value["accuracy"] for key, value in single_link.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()