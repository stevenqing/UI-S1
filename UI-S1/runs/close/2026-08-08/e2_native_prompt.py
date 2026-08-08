import hashlib
import json
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    e1_path = RUN_DIR / "e1_arm_aggregator_matrix.json"
    e1 = json.loads(e1_path.read_text())
    config = yaml.safe_load((RUN_DIR / "configs/native_adapters.yaml").read_text())
    if config["status"] != "FROZEN_BEFORE_NATIVE_INFERENCE_ASSET_RESTORE_PENDING":
        raise ValueError("native adapter config is not frozen")
    if not e1["E_K1"]:
        raise RuntimeError(
            "E1 allows E2, but native inference is intentionally not implemented in the cancellation finalizer"
        )
    result = {
        "schema_version": 1,
        "status": "CANCELLED_NOT_RUN",
        "reason": "E_K1_TRIGGERED",
        "anchor_inference_started": False,
        "four_arm_inference_started": False,
        "sota_claim": "OPEN_NOT_EVALUATED",
        "androidcontrol_decision": "CANCELLED_BY_E_K1",
        "e1_sha256": sha256_file(e1_path),
        "native_adapter_config_sha256": sha256_file(RUN_DIR / "configs/native_adapters.yaml"),
        "note": "E1 failed the preregistered majority primary gate on both Mind2Web and ScreenSpot-Pro. Per SPEC, no native-prompt GPU inference is launched and paused AndroidControl lanes are not resumed."
    }
    output = RUN_DIR / "e2_native_prompt.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
