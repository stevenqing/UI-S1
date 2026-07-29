import hashlib
import json
from pathlib import Path

from infer import DATASET_REVISION, MODEL_REVISIONS, OFFICIAL_REPO_REVISION, RUN_DIR


MODEL_DIRS = {
    "KDEGroup/UI-AGILE-3B": RUN_DIR / "models/UI-AGILE-3B",
    "KDEGroup/UI-AGILE": RUN_DIR / "models/UI-AGILE-7B",
    "LZXzju/Qwen2.5-VL-3B-UI-R1-E": RUN_DIR / "models/UI-R1-E-3B",
    "ritzzai/GUI-R1:GUI-R1-3B": RUN_DIR / "models/GUI-R1/GUI-R1-3B",
    "ritzzai/GUI-R1:GUI-R1-7B": RUN_DIR / "models/GUI-R1/GUI-R1-7B",
}
DATA_FILES = {
    "low": RUN_DIR / "data/UI-AGILE-Data/android_control/androidcontrol_low_test.parquet",
    "high": RUN_DIR / "data/UI-AGILE-Data/android_control/androidcontrol_high_test.parquet",
}
SOURCE_FILES = (
    RUN_DIR / "repo/eval/android_control/inference_android_control.py",
    RUN_DIR / "repo/eval/android_control/eval.py",
    RUN_DIR / "repo/eval/android_control/utils.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    models = {}
    for model_name, model_dir in MODEL_DIRS.items():
        index_path = model_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
        actual_names = sorted(path.name for path in model_dir.glob("model-*.safetensors"))
        if shard_names != actual_names:
            raise ValueError(f"checkpoint index mismatch for {model_name}")
        models[model_name] = {
            "revision": MODEL_REVISIONS[model_name],
            "index_sha256": sha256_file(index_path),
            "shards": {name: sha256_file(model_dir / name) for name in shard_names},
        }
    result = {
        "status": "DOWNLOADED_HASH_INDEX_VERIFIED",
        "official_repo_revision": OFFICIAL_REPO_REVISION,
        "official_source_sha256": {
            str(path.relative_to(RUN_DIR)): sha256_file(path) for path in SOURCE_FILES
        },
        "dataset_revision": DATASET_REVISION,
        "data_sha256": {setting: sha256_file(path) for setting, path in DATA_FILES.items()},
        "models": models,
    }
    output = RUN_DIR / "artifact_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "models": {name: len(value["shards"]) for name, value in models.items()},
        "data": result["data_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()