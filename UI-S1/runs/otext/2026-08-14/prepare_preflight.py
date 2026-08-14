import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/otext_prereg.yaml"
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify(path, record):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"OTEXT input drift: {path}")


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_BEFORE_ANY_OTEXT_OCR_OR_LABEL_STATISTIC":
        raise PermissionError("OTEXT prereg mismatch")
    upstream_paths = {
        "spec": RUN_DIR / "SPEC.md",
        "orth_adjudication": ROOT / "runs/orth/2026-08-14/ORTH_ADJUDICATION.json",
        "orth_decision": ROOT / "runs/orth/2026-08-14/SCOPING_DECISION.json",
        "split_preflight": ROOT / "runs/split/2026-08-14/PREFLIGHT.json",
        "cev_main": ROOT / "runs/cev/2026-08-09/cev_main.py",
        "e1_aggregator": ROOT / "runs/close/2026-08-08/e1_arm_aggregator_matrix.py",
        "sourcebias_common": ROOT / "runs/sourcebias/2026-08-03/sourcebias_common.py",
    }
    upstream = {name: file_record(path) for name, path in upstream_paths.items()}
    if upstream["spec"]["sha256"] != "d6cb245c800179c52b0312744b2eb6756358ed6b25b790950b1dd178e5d7f650":
        raise PermissionError("OTEXT spec hash mismatch")
    orth = json.loads(upstream_paths["orth_adjudication"].read_text())
    decision = json.loads(upstream_paths["orth_decision"].read_text())
    if orth.get("status") != "COMPLETE_SCOPING" or decision.get("direction") != "PREREGISTER_OCR_CONFIRMATORY":
        raise PermissionError("OTEXT ORTH boundary mismatch")
    split = json.loads(upstream_paths["split_preflight"].read_text())
    if split.get("screenspot_rows") != 1581:
        raise PermissionError("OTEXT image manifest mismatch")
    for value in split["images"].values():
        verify(ROOT / value["path"], value)
    easy_root = Path.home() / ".EasyOCR/model"
    rapid_root = Path(sys.prefix) / "lib/python3.12/site-packages/rapidocr_onnxruntime"
    assets = {
        "easyocr_craft": file_record(easy_root / "craft_mlt_25k.pth"),
        "easyocr_english": file_record(easy_root / "english_g2.pth"),
        "rapidocr_config": file_record(rapid_root / "config.yaml"),
        "rapidocr_det": file_record(rapid_root / "models/ch_PP-OCRv4_det_infer.onnx"),
        "rapidocr_rec": file_record(rapid_root / "models/ch_PP-OCRv4_rec_infer.onnx"),
        "rapidocr_cls": file_record(rapid_root / "models/ch_ppocr_mobile_v2.0_cls_infer.onnx"),
    }
    result = {
        "schema_version": 1,
        "status": "PASS_OTEXT_PREFLIGHT_POST_SELECTION_VALIDATION",
        "evidence_status": "POST_SELECTION_VALIDATION",
        "confirmatory_claim_allowed": False,
        "gpu_used": False,
        "ocr_forward_started": False,
        "stage0_label_statistics_computed": False,
        "stage1_heldout_opened": False,
        "upstream": upstream,
        "images": {
            "rows": 1581, "all_reverified": True,
            "total_bytes": sum(value["bytes"] for value in split["images"].values()),
            "source_manifest": upstream["split_preflight"],
        },
        "engines": {
            "primary": "easyocr",
            "replication": "rapidocr_onnxruntime",
            "easyocr_version": importlib.metadata.version("easyocr"),
            "rapidocr_version": importlib.metadata.version("rapidocr-onnxruntime"),
            "opencv_version": importlib.metadata.version("opencv-python-headless"),
            "assets": assets,
            "raw_must_be_regenerated": True,
        },
        "environment": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pillow": importlib.metadata.version("Pillow"),
            "pyyaml": importlib.metadata.version("PyYAML"),
        },
        "retention_extension": {
            "dataset_snapshot_manifest_required": True,
            "external_inputs_required": True,
            "raw_OCR_required": True,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "images": result["images"], "engines": {k: result["engines"][k] for k in ("primary", "replication", "easyocr_version", "rapidocr_version")}}, indent=2))


if __name__ == "__main__":
    main()