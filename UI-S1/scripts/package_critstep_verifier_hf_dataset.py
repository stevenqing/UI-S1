#!/usr/bin/env python3
"""Package critical-step verifier data for Hugging Face upload.

The local LLaMA-Factory files use absolute screenshot paths. This creates a
portable dataset folder with relative image paths and hard-linked/copied image
assets so the package can be trained from any checkout/download location.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_SOURCE_DIR = "outputs/critstep_verifier"
DEFAULT_OUTPUT_DIR = "outputs/critstep_verifier_hf_dataset"
IMAGE_ANCHOR = "gui360_history_arm_images/original_template"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def relative_image_path(image_path: str) -> str:
    path = Path(image_path)
    parts = path.parts
    anchor_parts = Path(IMAGE_ANCHOR).parts
    for idx in range(0, len(parts) - len(anchor_parts) + 1):
        if parts[idx : idx + len(anchor_parts)] == anchor_parts:
            suffix = Path(*parts[idx + len(anchor_parts) :])
            return str(Path("images") / suffix)
    if len(parts) >= 3:
        return str(Path("images") / path.name)
    raise ValueError(f"cannot make relative image path for {image_path}")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rewrite_sft_file(src_path: Path, dst_path: Path, image_map: Dict[str, str]) -> Tuple[int, int]:
    rows = read_json(src_path)
    for row in rows:
        new_images = []
        for image in row.get("images", []):
            rel = relative_image_path(str(image))
            image_map[str(image)] = rel
            new_images.append(rel)
        row["images"] = new_images
    write_json(dst_path, rows)
    unique_images = {image for row in rows for image in row.get("images", [])}
    return len(rows), len(unique_images)


def rewrite_jsonl_file(src_path: Path, dst_path: Path, image_map: Dict[str, str]) -> Tuple[int, int]:
    rows = read_jsonl(src_path)
    for row in rows:
        if "images" in row:
            row["images"] = [relative_image_path(str(image)) for image in row.get("images", [])]
        if "messages" in row:
            pass
    write_jsonl(dst_path, rows)
    unique_images = {image for row in rows for image in row.get("images", [])}
    return len(rows), len(unique_images)


def copy_known_artifacts(source_dir: Path, output_dir: Path) -> None:
    for name in [
        "manifest.json",
        "dataset_info.json",
        "dataset_info.snippet.json",
        "per_step.jsonl",
        "verifier_eval.md",
        "train_full_action_verifier_lora.yaml",
        "train_full_action_verifier_lora_step40.yaml",
    ]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    extra_sources = {
        "train_pool_per_step.jsonl": Path("outputs/critstep_elicit_train/per_step.jsonl"),
        "train_pool_summary.json": Path("outputs/critstep_elicit_train/summary.json"),
        "train_pool_decomposition.md": Path("outputs/critstep_elicit_train/decomposition.md"),
        "scope_report.md": Path("outputs/critstep_scope/scope.md"),
        "uia_element_selection_report.md": Path("outputs/critstep_reward_structure_uia/uia_element_selection_report.md"),
    }
    for dst_name, src in extra_sources.items():
        if src.exists():
            shutil.copy2(src, output_dir / dst_name)


def write_readme(output_dir: Path, stats: Dict[str, Any]) -> None:
    lines = [
        "# GUI-360 Critical-Step Full-Action Verifier Data",
        "",
        "No-leakage data package for training a generative verifier that judges full GUI actions from sampled candidates.",
        "",
        "## Splits",
        "",
        "| file | rows | unique images |",
        "|---|---:|---:|",
    ]
    for key in ["train_sft", "val_sft", "eval_slice_sft", "eval_slice_examples"]:
        item = stats.get(key, {})
        lines.append(f"| `{item.get('file')}` | {item.get('rows', 0)} | {item.get('unique_images', 0)} |")
    lines.extend([
        "",
        "## Leakage Policy",
        "",
        "Prompts contain only instruction, screenshot, action history, candidate action, and candidate UIA metadata.",
        "Matcher verdicts, GT actions, sample rank, frequency, success counts, and candidate source are metadata only and are not included in prompt text.",
        "",
        "## Training",
        "",
        "Use `dataset_info.json` with LLaMA-Factory and the included `train_full_action_verifier_lora.yaml` as the full overnight training config.",
        "The `eval_slice_sft.json` split is held-out and should not be used for training.",
        "",
        "## Contents",
        "",
        "- `train_sft.json`, `val_sft.json`: verifier SFT splits",
        "- `eval_slice_sft.json`, `eval_slice_examples.jsonl`, `per_step.jsonl`: held-out 200-step eval slice",
        "- `images/`: required screenshots referenced by relative paths",
        "- `train_pool_per_step.jsonl`: TRAIN-side sampled candidate pool used to build SFT data",
        "- `manifest.json`: construction manifest and counts",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_map: Dict[str, str] = {}
    stats: Dict[str, Any] = {}
    for stem, filename in [
        ("train_sft", "train_sft.json"),
        ("val_sft", "val_sft.json"),
        ("eval_slice_sft", "eval_slice_sft.json"),
    ]:
        rows, unique = rewrite_sft_file(source_dir / filename, output_dir / filename, image_map)
        stats[stem] = {"file": filename, "rows": rows, "unique_images": unique}
    rows, unique = rewrite_jsonl_file(source_dir / "eval_slice_examples.jsonl", output_dir / "eval_slice_examples.jsonl", image_map)
    stats["eval_slice_examples"] = {"file": "eval_slice_examples.jsonl", "rows": rows, "unique_images": unique}

    copied = 0
    for src_image, rel_image in sorted(image_map.items(), key=lambda item: item[1]):
        src = Path(src_image)
        if not src.exists():
            raise FileNotFoundError(src)
        dst = output_dir / rel_image
        link_or_copy(src, dst)
        copied += 1
    stats["images"] = {"unique_source_images": len(image_map), "files_materialized": copied}
    copy_known_artifacts(source_dir, output_dir)
    write_readme(output_dir, stats)
    write_json(output_dir / "package_manifest.json", stats)
    print(json.dumps({"output_dir": str(output_dir), **stats}, indent=2), flush=True)


if __name__ == "__main__":
    main()