import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import transformers


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "repos" / "ShowUI"), *args], text=True
    ).strip()


def main() -> None:
    model_weight = ROOT / "models" / "ShowUI-2B" / "pytorch_model.bin"
    metadata = ROOT / "data" / "Mind2Web" / "metadata" / "hf_test_task.json"
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    provenance = {
        "baseline": "ShowUI-ZS",
        "benchmark": "Mind2Web",
        "split": "test_task",
        "source": {
            "repository": git_output("remote", "get-url", "origin"),
            "revision": git_output("rev-parse", "HEAD"),
            "clean": git_output("status", "--porcelain") == "",
        },
        "model": {
            "repository": "showlab/ShowUI-2B",
            "revision": "cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60",
            "weight_bytes": model_weight.stat().st_size,
            "weight_sha256": sha256(model_weight),
        },
        "processor": {
            "repository": "Qwen/Qwen2-VL-2B-Instruct",
            "revision": "895c3a49bc3fa70a340399125c650a463535e71c",
        },
        "data": {
            **manifest,
            "metadata_sha256_recomputed": sha256(metadata),
        },
        "configuration": {
            "num_history": 2,
            "interleaved_history": "tttt",
            "min_visual_tokens": 256,
            "max_visual_tokens": 1344,
            "uigraph_test": False,
            "lm_skip_ratio": 0,
            "max_new_tokens": 128,
            "generation": "greedy/default released generate path",
            "precision": "bfloat16",
            "num_shards": 4,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        },
    }
    if provenance["source"]["revision"] != "21ed7cb24be0cc877bb8352ee34d58a9aea2c876":
        raise ValueError("ShowUI source revision drifted")
    if provenance["model"]["weight_sha256"] != "68080df785764e98976eb9cc93a07c6c69cf8a6933738496e02aef55b53d2aa3":
        raise ValueError("ShowUI model weight hash mismatch")
    if manifest["metadata_sha256"] != provenance["data"]["metadata_sha256_recomputed"]:
        raise ValueError("prepared metadata hash mismatch")
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
