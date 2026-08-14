import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
OUTPUT_PATH = RUN_DIR / "ARM2.json"


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    value = preflight["mind2web_dom_ax"]
    if value["status"] != "HISTORICALLY_AVAILABLE_CURRENTLY_MISSING":
        raise PermissionError("ORTH Arm 2 availability branch changed")
    result = {
        "schema_version": 1,
        "status": "FULL_DOM_AX_UNAVAILABLE_HISTORICAL_DATA_CURRENTLY_MISSING",
        "historical_evidence": {
            "dataset_revision": value["historical_dataset_revision"],
            "actions": value["historical_actions"],
            "episodes": value["historical_episodes"],
            "complete_audit": value["historical_complete_audit"],
        },
        "current_xfer_lane": {
            "rows": value["xfer_rows"],
            "full_tree_field": value["xfer_has_full_tree_field"],
            "positive_snippet_rows": value["xfer_positive_snippet_rows"],
            "positive_snippets_are_label_selected": True,
        },
        "missing_paths": [
            path for path, exists in zip(
                value["current_expected_paths"], value["current_expected_paths_exist"]
            ) if not exists
        ],
        "prediction_metrics_computed": False,
        "reason": "complete_2094_action_HTML_lane_was_historically_audited_but_local_dataset_and_scores_are_absent;_2080_XFER_lane_only_retains_GT_positive_snippets",
        "setting_boundary": "DOM_or_AX_would_change_the_pure_screenshot_setting_and_require_a_separate_confirmatory_protocol",
        "next_design_requirement": "restore_and_hash_the_official_dataset_before_any_DOM_predictor_scoping;_never_use_pos_candidates_as_predictor_input",
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()