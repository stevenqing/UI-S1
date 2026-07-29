import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "UI-TARS-72B-SFT"
MANIFEST = ROOT / "uitars72_checkpoint_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    shards = sorted(MODEL.glob("model-*.safetensors"))
    referenced = {MODEL / name for name in index["weight_map"].values()}
    if len(shards) != 64 or set(shards) != referenced:
        raise ValueError("checkpoint shard coverage does not match the index")
    if sum(path.stat().st_size for path in shards) != manifest["local_safetensors_bytes"]:
        raise ValueError("checkpoint byte count mismatch")
    if index["metadata"]["total_size"] != manifest["index_tensor_bytes"]:
        raise ValueError("index tensor byte count mismatch")
    if sha256(MODEL / "model.safetensors.index.json") != manifest["index_sha256"]:
        raise ValueError("index hash mismatch")
    if sha256(MODEL / "config.json") != manifest["config_sha256"]:
        raise ValueError("config hash mismatch")
    manifest["shard_sha256"] = [sha256(path) for path in shards]
    manifest["status"] = "DOWNLOADED_HASH_INDEX_VERIFIED"
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "shards": len(shards)}, indent=2))


if __name__ == "__main__":
    main()