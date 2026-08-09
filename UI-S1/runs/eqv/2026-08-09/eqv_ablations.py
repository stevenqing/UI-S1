import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


def main():
    main_result = json.loads((RUN_DIR / "eqv_main.json").read_text())
    self_check = main_result["ABL4_self_check"]
    if not self_check["U_K4"]:
        raise ValueError("full ablation runner is only finalized here for the U-K4 fail-closed path")
    result = {
        "schema_version": 1,
        "status": "PARTIAL_STOPPED_BY_U_K4",
        "ABL4": {
            "status": "FAIL_U_K4",
            "main_variant": "complete_link_with_lineage_dedup",
            "main_accuracy": self_check["EQV_ABL4_accuracy"],
            "A2_accuracy": self_check["A2_accuracy"],
            "main_minus_A2": self_check["EQV_ABL4_minus_A2"],
            "debug_factorial": self_check["debug_diagnostics"],
            "diagnosis": "complete_link_candidate_votes_matches_A2_point_accuracy_exactly_but_lineage_dedup_reduces_accuracy_by_more_than_MDE",
        },
        "ABL1": {
            "status": "RUN_ONLY_AS_U_K4_DEBUG",
            "result": self_check["debug_diagnostics"]["complete_candidate_votes"],
        },
        "ABL2": {
            "status": "RUN_ONLY_AS_U_K4_DEBUG",
            "with_lineage_dedup": self_check["debug_diagnostics"]["single_lineage_dedup"],
            "with_candidate_votes": self_check["debug_diagnostics"]["single_candidate_votes"],
        },
        "ABL3": {"status": "CANCELLED_NOT_RUN_BY_U_K4"},
        "ABL5": {"status": "CANCELLED_NOT_RUN_BY_U_K4"},
        "thresholds_changed_after_results": False,
        "primary_adjudication_permitted": False,
    }
    (RUN_DIR / "eqv_ablations.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
