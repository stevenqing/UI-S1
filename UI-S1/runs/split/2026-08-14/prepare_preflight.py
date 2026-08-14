import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
H1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"
MODEL_SPECS = {
    "Qwen3-VL-8B-Instruct": {
        "path": ROOT / "runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "required": True,
    },
    "GTA1-7B": {
        "path": ROOT / "runs/collision-law/2026-07-30/w3_assets/GTA1-7B",
        "revision": "701bedc80b447863bd60e3318ae44f6cbbfafd78",
        "required": True,
    },
    "Qwen2.5-VL-7B-Instruct": {
        "path": ROOT / "runs/mind2web-tongui/2026-07-28/models/Qwen2.5-VL-7B-Instruct",
        "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "required": False,
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    return {
        "path": Path(path).relative_to(ROOT).as_posix(),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256_file(path),
    }


def image_manifest():
    rows = {}
    for shard in sorted(H1_ROOT.glob("shard-*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id in rows:
                raise ValueError(f"SPLIT duplicate H1 identity: {row_id}")
            rows[row_id] = {
                "img_filename": str(row["img_filename"]),
                "img_size": [int(value) for value in row["img_size"]],
                "instruction_sha256": hashlib.sha256(
                    str(row["instruction"]).encode()
                ).hexdigest(),
            }
    if len(rows) != 1581:
        raise ValueError(f"SPLIT requires 1,581 image identities, found {len(rows)}")
    images = {}
    for row_id, row in sorted(rows.items()):
        path = DATA_ROOT / "images" / row["img_filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        images[row_id] = {
            **file_record(path),
            "img_filename": row["img_filename"],
            "declared_size": row["img_size"],
            "instruction_sha256": row["instruction_sha256"],
        }
    return images


def model_manifest():
    output = {}
    for model_id, spec in MODEL_SPECS.items():
        path = spec["path"]
        index_path = path / "model.safetensors.index.json"
        if not index_path.is_file():
            if spec["required"]:
                raise FileNotFoundError(index_path)
            output[model_id] = {
                "status": "DEFERRED_CHECKPOINT_MISSING",
                "historical_path": path.relative_to(ROOT).as_posix(),
                "historical_revision": spec["revision"],
                "required_for_primary": False,
            }
            continue
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"SPLIT model has no safetensors: {path}")
        output[model_id] = {
            "status": "READY",
            "path": path.relative_to(ROOT).as_posix(),
            "revision": spec["revision"],
            "index": file_record(index_path),
            "shards": [file_record(shard) for shard in shards],
            "total_weight_bytes": sum(shard.stat().st_size for shard in shards),
            "required_for_primary": spec["required"],
        }
    return output


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    images = image_manifest()
    models = model_manifest()
    gran_manifest_path = ROOT / "runs/gran/2026-08-14/INPUT_MANIFEST.json"
    result = {
        "schema_version": 1,
        "status": "PASS_SPLIT_PREFLIGHT_QWEN3_GTA1_READY_QWEN25_DEFERRED",
        "gpu_used": False,
        "label_statistics_computed": False,
        "Delta2_computed": False,
        "probe_forward_started": False,
        "screenspot_rows": len(images),
        "images": images,
        "models": models,
        "gran_input_manifest": file_record(gran_manifest_path),
        "environment": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "vllm": importlib.metadata.version("vllm"),
            "pillow": importlib.metadata.version("Pillow"),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "pyyaml": importlib.metadata.version("PyYAML"),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if json.loads(OUTPUT_PATH.read_text()) != result:
        raise ValueError("SPLIT preflight readback mismatch")
    print(json.dumps({
        "status": result["status"],
        "screenspot_rows": result["screenspot_rows"],
        "models": {
            model: value["status"] for model, value in models.items()
        },
        "image_bytes": sum(value["bytes"] for value in images.values()),
        "model_weight_bytes": {
            model: value.get("total_weight_bytes", 0)
            for model, value in models.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()