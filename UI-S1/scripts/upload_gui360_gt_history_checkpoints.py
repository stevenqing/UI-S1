#!/usr/bin/env python3
"""Upload GUI-360 gt-history checkpoints to Hugging Face Hub.

By default this uploads the inference-ready model artifacts for checkpoint-13
and checkpoint-26, excluding DeepSpeed optimizer states under global_step*/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_REPO_ID = "Stevenshuqing/gui360-gt-history-full-sft"
DEFAULT_CHECKPOINT_ROOT = "train_GUI_360/llamafactory/output/gui360_gt_history_full_sft"
DEFAULT_CHECKPOINTS = ["checkpoint-13", "checkpoint-26"]
EVAL_LOSSES = {
    "checkpoint-13": "0.18081653118133545",
    "checkpoint-26": "0.12538108229637146",
    "checkpoint-39": "0.11632098257541656",
    "checkpoint-52": "0.11428435146808624",
}
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.jinja",
    "*.txt",
    "*.bin",
    "README.md",
]
IGNORE_PATTERNS = [
    "global_step*/**",
    "rng_state_*.pth",
    "scheduler.pt",
    "optimizer.pt",
    "zero_to_fp32.py",
]


def write_checkpoint_readme(path: Path, *, repo_id: str, checkpoints: list[str]) -> None:
    checkpoint_lines = []
    for name in checkpoints:
        eval_loss = EVAL_LOSSES.get(name, "not recorded")
        checkpoint_lines.append(f"- `{name}`: eval_loss `{eval_loss}`")

    text = f"""---
library_name: transformers
base_model: Qwen/Qwen2.5-VL-7B-Instruct
tags:
- gui360
- qwen2.5-vl
- sft
- history
---

# GUI-360 GT-History Full SFT Checkpoint

This repository stores epoch checkpoints for the GUI-360 gt-history full-parameter SFT arm.

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Dataset arm: `gt_history`
- Training recipe: LLaMA-Factory full-parameter SFT + DeepSpeed ZeRO-3
- Local source repo: `UI-S1`
- Hub repo: `{repo_id}`

The uploaded folders contain inference-ready model artifacts only. DeepSpeed optimizer states under `global_step*/` are intentionally excluded to keep the Hub artifact compact.

## Checkpoints

{chr(10).join(checkpoint_lines)}

"""
    path.write_text(text, encoding="utf-8")


def checkpoint_sort_key(name: str) -> int:
    try:
        return int(name.rsplit("-", maxsplit=1)[1])
    except (IndexError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload GUI-360 gt-history checkpoints to Hugging Face Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--readme-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoints = args.checkpoint or DEFAULT_CHECKPOINTS

    root = Path(args.checkpoint_root)
    missing = [name for name in checkpoints if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint dirs under {root}: {missing}")
    readme_checkpoints = sorted(
        {name for name in EVAL_LOSSES if (root / name).is_dir()} | set(checkpoints),
        key=checkpoint_sort_key,
    )

    print(f"repo_id={args.repo_id}")
    print(f"checkpoint_root={root}")
    for name in checkpoints:
        print(f"will_upload={root / name} -> {name}/")
    print(f"readme_checkpoints={','.join(readme_checkpoints)}")
    if args.dry_run:
        return

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    readme = root / "README.md"
    write_checkpoint_readme(readme, repo_id=args.repo_id, checkpoints=readme_checkpoints)
    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add GUI-360 gt-history checkpoint README",
    )
    if args.readme_only:
        print("uploaded README.md")
        return

    for name in checkpoints:
        api.upload_folder(
            folder_path=str(root / name),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="model",
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"Upload {name} inference checkpoint",
        )
        print(f"uploaded {name}")


if __name__ == "__main__":
    main()