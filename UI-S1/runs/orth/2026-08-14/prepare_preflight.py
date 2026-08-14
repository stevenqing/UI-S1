import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"
SPEC_PATH = RUN_DIR / "SPEC.md"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_record(path, value):
    path = Path(path)
    if path.stat().st_size != value["bytes"] or sha256_file(path) != value["sha256"]:
        raise ValueError(f"ORTH input drift: {path}")


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    mask_preflight_path = ROOT / "runs/mask/2026-08-14/PREFLIGHT.json"
    mask_preflight = json.loads(mask_preflight_path.read_text())
    if mask_preflight.get("screenspot_rows") is not None:
        raise ValueError("ORTH expected compact MASK preflight schema")
    split_preflight_path = ROOT / "runs/split/2026-08-14/PREFLIGHT.json"
    split_preflight = json.loads(split_preflight_path.read_text())
    if split_preflight.get("screenspot_rows") != 1581:
        raise PermissionError("ORTH SSPro image manifest mismatch")
    for value in split_preflight["images"].values():
        verify_record(ROOT / value["path"], value)

    ceil_manifest_path = ROOT / "runs/ceil/2026-08-14/ARM_B_MANIFEST.json"
    ceil_manifest = json.loads(ceil_manifest_path.read_text())
    if ceil_manifest.get("status") != "LOCKED_CEIL_ARM_B_OUTPUTS":
        raise PermissionError("ORTH CEIL Arm B manifest mismatch")
    for value in ceil_manifest["artifacts"].values():
        verify_record(ROOT / value["path"], value)

    easy_root = Path.home() / ".EasyOCR/model"
    rapid_root = Path(sys.prefix) / "lib/python3.12/site-packages/rapidocr_onnxruntime"
    engine_files = {
        "easyocr_craft": record(easy_root / "craft_mlt_25k.pth"),
        "easyocr_english_g2": record(easy_root / "english_g2.pth"),
        "rapidocr_config": record(rapid_root / "config.yaml"),
        "rapidocr_det": record(rapid_root / "models/ch_PP-OCRv4_det_infer.onnx"),
        "rapidocr_rec": record(rapid_root / "models/ch_PP-OCRv4_rec_infer.onnx"),
        "rapidocr_cls": record(rapid_root / "models/ch_ppocr_mobile_v2.0_cls_infer.onnx"),
    }

    mindact_root = ROOT / "runs/mindact/2026-07-29"
    historical_manifest = json.loads((mindact_root / "artifact_manifest.json").read_text())
    historical_audit = json.loads((mindact_root / "artifacts/full/audit.json").read_text())
    expected_dom_paths = [
        mindact_root / "data/Mind2Web",
        mindact_root / "data/source/scores_all_data.pkl",
    ]
    dom_available = all(path.exists() for path in expected_dom_paths)
    xfer_task_path = ROOT / "runs/xfer/2026-08-07/data/mind2web/mind2web_test_task.jsonl"
    task_rows = [json.loads(line) for line in xfer_task_path.read_text().splitlines() if line.strip()]
    if len(task_rows) != 2080:
        raise ValueError("ORTH M2W screenshot lane mismatch")
    task_schema = sorted(task_rows[0])
    step_schema = sorted(task_rows[0]["step"])
    has_full_tree_field = any(
        key in task_rows[0] or key in task_rows[0]["step"]
        for key in ("dom", "raw_html", "cleaned_html", "ax_tree", "accessibility_tree", "neg_candidates")
    )
    positive_snippet_rows = sum(bool(row["step"].get("pos_candidates")) for row in task_rows)

    inputs = {
        "spec": record(SPEC_PATH),
        "split_preflight": record(split_preflight_path),
        "mask_preflight": record(mask_preflight_path),
        "mask_stage1": record(ROOT / "runs/mask/2026-08-14/STAGE1.json"),
        "ceil_arm_b": record(ROOT / "runs/ceil/2026-08-14/ARM_B.json"),
        "ceil_arm_b_rows": record(ROOT / "runs/ceil/2026-08-14/ARM_B_ROWS.jsonl"),
        "ceil_arm_b_manifest": record(ceil_manifest_path),
        "sspro_h1_manifest": record(ROOT / "runs/gran/2026-08-14/INPUT_MANIFEST.json"),
        "m2w_tasks": record(xfer_task_path),
        "mindact_manifest": record(mindact_root / "artifact_manifest.json"),
        "mindact_audit": record(mindact_root / "artifacts/full/audit.json"),
    }
    result = {
        "schema_version": 1,
        "status": "PASS_ORTH_PREFLIGHT_CPU_OCR_READY",
        "gpu_used": False,
        "ocr_forward_started": False,
        "orth_statistics_computed": False,
        "inputs": inputs,
        "screenspot": {
            "rows": 1581,
            "images_all_reverified": True,
            "text_rows": 977,
            "icon_rows": 604,
            "image_bytes": sum(value["bytes"] for value in split_preflight["images"].values()),
        },
        "ocr_engines": {
            "easyocr": {
                "version": importlib.metadata.version("easyocr"),
                "device": "cpu",
                "languages": ["en"],
                "model_files": ["easyocr_craft", "easyocr_english_g2"],
                "call": {
                    "detail": 1, "paragraph": False, "decoder": "greedy",
                    "beamWidth": 5, "batch_size": 1, "workers": 0,
                    "min_size": 10, "rotation_info": None,
                    "text_threshold": 0.7, "low_text": 0.4,
                    "link_threshold": 0.4, "canvas_size": 2560,
                    "mag_ratio": 1.0,
                },
            },
            "rapidocr_onnxruntime": {
                "version": importlib.metadata.version("rapidocr-onnxruntime"),
                "device": "cpu",
                "model_files": ["rapidocr_config", "rapidocr_det", "rapidocr_rec", "rapidocr_cls"],
                "call": {"use_det": True, "use_cls": True, "use_rec": True},
            },
            "opencv_python_headless": importlib.metadata.version("opencv-python-headless"),
            "model_assets": engine_files,
        },
        "mind2web_dom_ax": {
            "status": (
                "FULL_DOM_DATA_CURRENTLY_AVAILABLE"
                if dom_available else "HISTORICALLY_AVAILABLE_CURRENTLY_MISSING"
            ),
            "historical_dataset_revision": historical_manifest["dataset"]["revision"],
            "historical_actions": historical_audit["actions"],
            "historical_episodes": historical_audit["episodes"],
            "historical_complete_audit": historical_audit["coverage"] == "COMPLETE",
            "current_expected_paths": [str(path) for path in expected_dom_paths],
            "current_expected_paths_exist": [path.exists() for path in expected_dom_paths],
            "xfer_rows": len(task_rows),
            "xfer_top_schema": task_schema,
            "xfer_step_schema": step_schema,
            "xfer_has_full_tree_field": has_full_tree_field,
            "xfer_positive_snippet_rows": positive_snippet_rows,
            "positive_snippets_are_label_selected_and_not_predictor_input": True,
        },
        "environment": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pillow": importlib.metadata.version("Pillow"),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if json.loads(OUTPUT_PATH.read_text()) != result:
        raise ValueError("ORTH preflight readback mismatch")
    print(json.dumps({
        "status": result["status"],
        "ocr_engines": {key: value["version"] for key, value in result["ocr_engines"].items() if isinstance(value, dict) and "version" in value},
        "screenspot": result["screenspot"],
        "dom_ax": result["mind2web_dom_ax"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()