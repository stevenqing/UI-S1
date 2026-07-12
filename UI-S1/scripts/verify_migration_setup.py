#!/usr/bin/env python3
"""Verify a migrated UI-S1 research workspace without starting GPU jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


ENVIRONMENTS = {
    ".venv-qwen3-vllm": {
        "python": "3.11.15",
        "packages": {
            "torch": "2.8.0",
            "transformers": "4.57.1",
            "vllm": "0.11.0",
            "huggingface-hub": "0.36.2",
            "deepspeed": "0.16.9",
            "llamafactory": "0.9.5",
            "pyarrow": "24.0.0",
            "qwen-vl-utils": "0.0.14",
        },
    },
    ".venv-qwen35-vllm": {
        "python": "3.11.15",
        "packages": {
            "torch": "2.11.0",
            "transformers": "5.12.1",
            "vllm": "0.23.0",
            "huggingface-hub": "1.19.0",
            "flashinfer-python": "0.6.12",
        },
    },
}


MODELS = {
    "bridge": {
        "gui360-fullparam-sft-step250": {
            "architecture": "Qwen2_5_VLForConditionalGeneration",
            "weight_bytes": 16_584_414_544,
            "revision": "89a3556d0e3b38702deae86d1fa090b3eb4748d1",
        },
        "Qwen3-VL-8B-Instruct": {
            "architecture": "Qwen3VLForConditionalGeneration",
            "weight_bytes": 17_534_339_512,
            "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        },
        "Qwen3.5-9B": {
            "architecture": "Qwen3_5ForConditionalGeneration",
            "weight_bytes": 19_306_310_880,
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        },
        "llava-1.5-7b-hf": {
            "architecture": "LlavaForConditionalGeneration",
            "weight_bytes": 14_126_946_048,
            "revision": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
        },
    },
    "full": {
        "InternVL3-8B": {
            "architecture": "InternVLChatModel",
            "weight_bytes": 15_888_831_920,
            "revision": "853e3a797a661694b1b8ece0cb72dc2b23e3dac9",
        },
        "Qwen3.5-35B-A3B": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "weight_bytes": 71_903_878_016,
            "revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        },
        "Qwen2.5-VL-7B-Instruct": {
            "architecture": "Qwen2_5_VLForConditionalGeneration",
            "weight_bytes": 16_584_414_560,
            "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        },
    },
}


FILES = {
    "outputs/validation_2k/data/train_episodes.jsonl": "7af451fb32cd3df60c19a3f281c4b59cb574300519d0bb7d5e961d0bf9d6958e",
    "outputs/validation_2k/data/test_episodes.jsonl": "0f6fb7154e259eff9edd5e0cd59c7780293f71750b9a3b501af72c8860c258b5",
    "outputs/rl_feasibility/per_step.jsonl": "71ab0df74d5f25a8aba5a77cba15959ca3ae390b6dcc979d4178dc4d67e22cfc",
    "outputs/multiagent_complementarity/target_ids.json": "509327c7de49565e423afd9d8078631fe1df02a5c2af1440c2682e59870f71ad",
    "outputs/multiagent_complementarity/qwen3_vl_candidates.jsonl": "c3e4e0f6984ac5a106b5dc585ce584e1046e033a08ed4d37086df5fd9e63a6ca",
    "outputs/multiagent_complementarity/qwen35_candidates.jsonl": "aaf05b9cc36c081b69f48abcc4b6b7c26a1648fa2ca12fe7f84a40070d4fe08f",
    "outputs/multiagent_complementarity/extra_tiers/llava15_candidates.jsonl": "fa168f868559467eca220d49b11eebe9559bac6e64a76e4a39b3299d0a7f005e",
    "outputs/pass8_selector_study/frozen_v1/manifest.json": "a993d8d18b3ad997622dbd0503257aa624cb7715320729d73a76f888a28df89c",
    "outputs/multiagent_trajectory_revision/full_v1/causal_arms/a1_gt_target_gt_history.jsonl": "03865e32e0bac2b59177d7cf656bacba3837b5666655ec8078e1401fac2db046",
    "outputs/multiagent_trajectory_revision/full_v1/causal_arms/a5_revision_target_gt_history.jsonl": "f10bc370acd8997d830cd944497780d48f939d72e699a6f1b412c5a1a7fd8df3",
    "outputs/multiagent_trajectory_revision/full_v1/causal_eval/a5_gt_history_grid/merged.jsonl": "9e286d50434f68bb7c4a4520f898608a8ee62f9ea5d03a31f7b858f9f48421fd",
    "outputs/multiagent_trajectory_revision/full_v1/utility_gate/a13_oracle_student_rescue_gt_history.jsonl": "fb0e892e6df3d7edc894741dffac7ca3c85007ca9d314046d1b9bfa405352001",
    "outputs/multiagent_trajectory_revision/full_v1/utility_gate/a15_student_rescue25_replay75.jsonl": "7f47ec0eb2a9ae37d01a8af324242cf60aa391a367be1debdadfc427c220242b",
}


REQUIRED_PASS8_PATHS = (
    "outputs/pass8_selector_study/frozen_v1/blind/dev.jsonl",
    "outputs/pass8_selector_study/frozen_v1/blind/locked_test.jsonl",
    "outputs/pass8_selector_study/frozen_v1/sealed_labels/dev.jsonl",
    "outputs/pass8_selector_study/frozen_v1/sealed_labels/locked_test.jsonl",
    "outputs/pass8_selector_study/runtime/selectors/current/locked_test.jsonl",
    "outputs/pass8_selector_study/runtime/selectors/strong/locked_test.jsonl",
    "outputs/pass8_selector_study/runtime/selectors/cross_source_consensus/locked_test.jsonl",
    "outputs/pass8_selector_study/eval/locked_test/current_per_step.jsonl",
    "outputs/pass8_selector_study/eval/locked_test/cross_source_consensus_per_step.jsonl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_json(command: list[str], cwd: Path) -> Any:
    output = subprocess.check_output(command, cwd=cwd, text=True)
    return json.loads(output)


def verify_environment(root: Path, name: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    python = root / name / "bin" / "python"
    if not python.exists():
        return {"ok": False, "error": "missing python executable"}
    code = (
        "import importlib.metadata as m,json,platform;"
        f"names={list(expected['packages'])!r};"
        "print(json.dumps({'python':platform.python_version(),'packages':{n:m.version(n) for n in names}}))"
    )
    try:
        actual = run_json([str(python), "-c", code], root)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"environment query failed: {exc}"}
    mismatches = {}
    if actual["python"] != expected["python"]:
        mismatches["python"] = {"expected": expected["python"], "actual": actual["python"]}
    for package, version in expected["packages"].items():
        if actual["packages"].get(package) != version:
            mismatches[package] = {"expected": version, "actual": actual["packages"].get(package)}
    return {"ok": not mismatches, "actual": actual, "mismatches": mismatches}


def local_revision(model_dir: Path) -> str | None:
    metadata = model_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not metadata.exists():
        return None
    return metadata.read_text(encoding="utf-8").splitlines()[0].strip()


def verify_model(root: Path, name: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    model_dir = root / "checkpoints" / name
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {"ok": False, "error": "missing config.json"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = list(config.get("architectures") or [])
    shards = list(model_dir.glob("*.safetensors"))
    weight_bytes = sum(path.stat().st_size for path in shards)
    index_path = model_dir / "model.safetensors.index.json"
    missing_shards: list[str] = []
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        missing_shards = sorted(set(index.get("weight_map", {}).values()) - {path.name for path in model_dir.iterdir()})
    actual = {
        "architectures": architectures,
        "weight_bytes": weight_bytes,
        "revision": local_revision(model_dir),
        "missing_index_shards": missing_shards,
    }
    mismatches = {}
    if expected["architecture"] not in architectures:
        mismatches["architecture"] = {"expected": expected["architecture"], "actual": architectures}
    for field in ("weight_bytes", "revision"):
        if actual[field] != expected[field]:
            mismatches[field] = {"expected": expected[field], "actual": actual[field]}
    if missing_shards:
        mismatches["missing_index_shards"] = missing_shards
    return {"ok": not mismatches, "actual": actual, "mismatches": mismatches}


def verify_file(root: Path, relative: str, expected_hash: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        return {"ok": False, "error": "missing"}
    actual_hash = sha256(path)
    return {"ok": actual_hash == expected_hash, "expected_sha256": expected_hash, "actual_sha256": actual_hash, "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--profile", choices=("bridge", "full"), default="bridge")
    parser.add_argument("--skip-env", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    results: dict[str, Any] = {"root": str(root), "profile": args.profile}
    if not args.skip_env:
        results["environments"] = {name: verify_environment(root, name, expected) for name, expected in ENVIRONMENTS.items()}
    model_specs = dict(MODELS["bridge"])
    if args.profile == "full":
        model_specs.update(MODELS["full"])
    results["models"] = {name: verify_model(root, name, expected) for name, expected in model_specs.items()}
    results["files"] = {relative: verify_file(root, relative, expected_hash) for relative, expected_hash in FILES.items()}
    results["pass8_paths"] = {relative: (root / relative).exists() for relative in REQUIRED_PASS8_PATHS}
    image_counts = {}
    for split, expected in (("train", 12_574), ("test", 7_498)):
        directory = root / "outputs" / "validation_2k" / "data" / "images" / split
        actual = sum(1 for path in directory.rglob("*") if path.is_file()) if directory.exists() else 0
        image_counts[split] = {"ok": actual == expected, "expected": expected, "actual": actual}
    results["image_counts"] = image_counts

    failures = []
    for section in ("environments", "models", "files"):
        for name, item in results.get(section, {}).items():
            if not item.get("ok"):
                failures.append(f"{section}:{name}")
    failures.extend(f"pass8_paths:{name}" for name, exists in results["pass8_paths"].items() if not exists)
    failures.extend(f"image_counts:{name}" for name, item in image_counts.items() if not item["ok"])
    results["ok"] = not failures
    results["failures"] = failures
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if results["ok"] else 1)


if __name__ == "__main__":
    main()