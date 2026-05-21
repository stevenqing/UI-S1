#!/usr/bin/env python3
"""Upload representative checkpoints to HuggingFace Hub."""

import os
import json
from huggingface_hub import HfApi, create_repo

api = HfApi()
HUB_ORG = "Stevenshuqing"
BASE = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

CHECKPOINTS = [
    {
        "repo": f"{HUB_ORG}/gui360-fullparam-sft-step250",
        "path": f"{BASE}/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-250",
        "description": "Full-parameter SFT on GUI-360 balanced 2K data (Qwen2.5-VL-7B-Instruct). TSR=22.2%, StepSR=69.3%. Best overall. ~epoch 3.7, LLaMA-Factory+ZeRO-3.",
        "exclude_patterns": [
            "bf16_zero_pp_rank_",  # optimizer states (~86G)
            "rng_state_",          # random states
            "scheduler.pt",
            "global_step",
        ],
    },
    {
        "repo": f"{HUB_ORG}/gui360-cooperative-lora-svd-sft",
        "path": f"{BASE}/checkpoints/v15_peft_balanced_cooperative_r128",
        "description": "Cooperative LoRA (r=128, T=2 comm rounds) with SVD initialization from full SFT. TSR=18.6%, StepSR=65.3%. ~67M params. Starting checkpoint for RL.",
        "exclude_patterns": [],
    },
    {
        "repo": f"{HUB_ORG}/gui360-cooperative-lora-rl-step25",
        "path": f"{BASE}/checkpoints/v15_rl_from_svd_extract/epoch-0_step-25",
        "description": "Cooperative LoRA after RL (GSPO) training. TSR=20.8%, StepSR=67.9%. Best LoRA result, 94% of full SFT performance with ~142M params.",
        "exclude_patterns": [],
    },
]

MODEL_CARD_TEMPLATE = """---
tags:
  - gui-agent
  - qwen2.5-vl
  - cooperative-lora
  - gui-360
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-7B-Instruct
datasets:
  - Stevenshuqing/gui360-balanced
---

# {title}

{description}

## Base Model
- **Qwen2.5-VL-7B-Instruct**

## Training Data
- GUI-360 balanced 2K episodes (17,264 steps)
- Action types: click, type, swipe (balanced sampling)

## Evaluation (GUI-360 test 1K balanced)

| Metric | Value |
|--------|-------|
| TSR (Task Success Rate) | {tsr} |
| StepSR (Step Success Rate) | {stepsr} |
| Progress | {progress} |

## Full Ranking

| # | Method | TSR | Params |
|---|--------|----:|--------|
| 1 | Full-param SFT step-250 | 22.2% | 7.6B |
| 2 | **V15 Cooperative RL step-25** | **20.8%** | ~142M |
| 3 | PEFT Cooperative r=128 (SVD) | 18.6% | ~67M |
| 4 | PEFT Standard r=128 (SVD) | 18.1% | ~67M |
| 5 | Base model (zero-shot) | 2.4% | — |

## Citation

Part of the Cooperative LoRA research for GUI agents.
"""


def should_upload(filename, exclude_patterns):
    for pat in exclude_patterns:
        if pat in filename:
            return False
    return True


def upload_checkpoint(ckpt):
    repo_id = ckpt["repo"]
    local_path = ckpt["path"]
    excludes = ckpt["exclude_patterns"]

    print(f"\n{'='*60}")
    print(f"Uploading: {repo_id}")
    print(f"From: {local_path}")
    print(f"{'='*60}")

    # Create repo
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"  Repo created/exists: {repo_id}")
    except Exception as e:
        print(f"  Repo creation: {e}")

    # List files to upload
    files_to_upload = []
    for root, dirs, files in os.walk(local_path):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, local_path)
            if should_upload(f, excludes):
                size = os.path.getsize(full_path)
                files_to_upload.append((full_path, rel_path, size))

    total_size = sum(s for _, _, s in files_to_upload)
    print(f"  Files to upload: {len(files_to_upload)}, total: {total_size / 1e9:.1f}G")

    for full_path, rel_path, size in files_to_upload:
        print(f"  Uploading {rel_path} ({size / 1e6:.1f}M)...")
        api.upload_file(
            path_or_fileobj=full_path,
            path_in_repo=rel_path,
            repo_id=repo_id,
            repo_type="model",
        )

    # Upload model card
    card_kwargs = {"title": repo_id.split("/")[-1], "description": ckpt["description"]}
    if "fullparam" in repo_id:
        card_kwargs.update(tsr="22.2%", stepsr="69.3%", progress="35.3%")
    elif "svd-sft" in repo_id:
        card_kwargs.update(tsr="18.6%", stepsr="65.3%", progress="31.0%")
    elif "rl-step25" in repo_id:
        card_kwargs.update(tsr="20.8%", stepsr="67.9%", progress="34.5%")

    readme = MODEL_CARD_TEMPLATE.format(**card_kwargs)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Done: https://huggingface.co/{repo_id}")


def main():
    for ckpt in CHECKPOINTS:
        upload_checkpoint(ckpt)
    print("\n=== All uploads complete ===")


if __name__ == "__main__":
    main()
