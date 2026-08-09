import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


def write(name, scope):
    result = {
        "schema_version": 1,
        "status": "CANCELLED_NOT_RUN_BY_U_K4",
        "scope": scope,
        "reason": "ABL4 implementation self-check exceeded the frozen ScreenSpot MDE; primary adjudication paused before downstream diagnostics",
        "new_inference_started": False,
        "result_claim": None,
    }
    (RUN_DIR / name).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    main_result = json.loads((RUN_DIR / "eqv_main.json").read_text())
    if not main_result["ABL4_self_check"]["U_K4"]:
        raise ValueError("diagnostic cancellation requires U-K4")
    outputs = {
        "eqv_source_bias.json": write("eqv_source_bias.json", "B1_winner_source_distribution"),
        "eqv_action_strata.json": write("eqv_action_strata.json", "Mind2Web_CLICK_TYPE_SELECT"),
    }
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
