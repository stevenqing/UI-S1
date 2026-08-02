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
        "status": "UNAVAILABLE_NO_RELEASED_IMPLEMENTATION_OR_FIXED_BUDGET_TRACE",
        "baseline": {
            "name": "GMS: Generalist Scanner Meets Specialist Locator",
            "paper": "arXiv:2509.24133v1",
            "reported_abstract_accuracy": 0.357,
            "pipeline": ["hierarchical_3x3_search", "iterative_refinement", "cross_modal_verification", "multi_agent_consensus", "adaptive_resolution_fusion"],
            "code_repository": None,
            "fixed_forward_budget": None,
            "per_row_traces": None,
        },
        "reference_only": {
            "mixed_N12_M1": l1["evaluations"]["mixed"]["12"]["accuracy"]["M1_ccm"],
            "mixed_N12_pass_at_12": l1["evaluations"]["mixed"]["12"]["accuracy"]["pass_at_n"],
            "comparison_valid": False,
            "reason": "GMS reported accuracy has no auditable 12-forward mapping or reusable candidate trace",
        },
        "prediction_X4": "NOT_EVALUATED",
        "kill_conditions": {"X-K4": "NOT_EVALUATED"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()