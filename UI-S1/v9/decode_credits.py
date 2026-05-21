#!/usr/bin/env python3
"""v9: Offline decode + credit extraction for cooperative LoRA.

For each sample in a single-step training jsonl (e.g. ac_train_thought.jsonl):
  1. Build the user-turn prompt.
  2. Generate model output via the cooperative wrapper.
  3. Parse the predicted action.
  4. Parse the GT action from the assistant turn.
  5. Compute (coord_correct, action_correct, has_coord, r_g, r_a).
  6. Append to credit_cache_shard{shard_id}.jsonl.

Usage (single shard):
  python v9/decode_credits.py \\
      --base_model checkpoints/Qwen2.5-VL-7B-Instruct \\
      --coop_checkpoint checkpoints/cooperative_thought_v6_5_ac/coop_epoch4 \\
      --train_jsonl datasets/cooperative_thought_ac/ac_train_thought.jsonl \\
      --out_dir v9/credits/v6_5_ac_ep4

Multi-shard SLURM array:
  --shard_id $SLURM_ARRAY_TASK_ID --num_shards $SLURM_ARRAY_TASK_COUNT
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

# Make the repo root importable when run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.cooperative_trajectory_common import (  # noqa: E402
    cooperative_generate,
    load_cooperative_model,
)
from v9.credit_assignment import (  # noqa: E402
    credit_for_sample,
    parse_action_from_text,
)


# ── GT parsing from training jsonl ────────────────────────────────────

_ACTION_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)


def parse_gt_action(assistant_value: str) -> Optional[Dict[str, Any]]:
    """Pull the GT action JSON out of the assistant turn."""
    m = _ACTION_RE.search(assistant_value)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


# ── Build prompt from a single-turn sample ────────────────────────────

def build_user_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a training jsonl row into Qwen-format messages (user only).

    The training jsonl uses the cooperative_thought format:
        [{"from": "human", "value": "<image>\\n<system+task text>"},
         {"from": "assistant", "value": "<think>...</think>\\n<action>{...}</action>"}]
    We drop the assistant turn and feed only the user turn for generation.
    """
    convs = sample["conversations"]
    image_paths = sample.get("images", [])

    messages: List[Dict[str, Any]] = []
    img_idx = 0
    for conv in convs:
        role = conv["from"]
        text = conv["value"]
        if role == "system":
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": text}],
            })
        elif role in ("human", "user"):
            content = []
            if "<image>" in text:
                text_clean = text.replace("<image>\n", "").replace("<image>", "").strip()
                if img_idx < len(image_paths):
                    pil = Image.open(image_paths[img_idx]).convert("RGB")
                    content.append({"type": "image", "image": pil})
                    img_idx += 1
                content.append({"type": "text", "text": text_clean})
            else:
                content.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content})
        # Skip assistant turns — we want to generate them.
    return messages


def get_image_size(sample: Dict[str, Any]) -> Optional[tuple]:
    """Return the (width, height) of the GT image used for the action."""
    paths = sample.get("images", [])
    if not paths:
        return None
    try:
        with Image.open(paths[0]) as im:
            return im.size  # (W, H)
    except Exception:
        return None


def get_resized_size(processor, sample: Dict[str, Any]) -> Optional[tuple]:
    """Compute (resized_w, resized_h) by running the image through the processor.

    Qwen2.5-VL's processor applies smart_resize and reports image_grid_thw,
    where (h_patches, w_patches) * merge_size * patch_size = resized pixels.
    """
    paths = sample.get("images", [])
    if not paths:
        return None
    try:
        img = Image.open(paths[0]).convert("RGB")
        out = processor(images=[img], text=["dummy"], return_tensors="pt")
        thw = out["image_grid_thw"][0].tolist()  # [t, h, w] in patches
        # Qwen2.5-VL: patch_size=14, merge_size=2 → each grid cell = 28 px
        merge_size = getattr(processor.image_processor, "merge_size", 2)
        patch_size = getattr(processor.image_processor, "patch_size", 14)
        resized_h = thw[1] * merge_size * patch_size
        resized_w = thw[2] * merge_size * patch_size
        return (resized_w, resized_h)
    except Exception:
        return None


# ── Main loop ─────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"credit_cache_shard{args.shard_id}.jsonl")

    # Load samples.
    samples = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    n_total = len(samples)
    print(f"[shard {args.shard_id}] Loaded {n_total} samples from {args.train_jsonl}")

    # Shard.
    shard_indices = list(range(args.shard_id, n_total, args.num_shards))
    if args.max_samples > 0:
        shard_indices = shard_indices[:args.max_samples]
    print(f"[shard {args.shard_id}] Will process {len(shard_indices)} samples")

    # Load model.
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    print(f"[shard {args.shard_id}] Loading model on {device} ...")
    model, processor, model_device = load_cooperative_model(
        base_model_path=args.base_model,
        coop_checkpoint_path=args.coop_checkpoint,
        device=device,
    )

    # Resume support.
    done_indices = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_indices.add(json.loads(line)["sample_idx"])
                    except Exception:
                        pass
        print(f"[shard {args.shard_id}] Resume: {len(done_indices)} samples already done")

    fout = open(out_path, "a" if args.resume else "w")

    n_processed = 0
    n_parse_fail = 0
    t0 = time.time()
    for idx in shard_indices:
        if idx in done_indices:
            continue
        sample = samples[idx]

        # GT action.
        assistant_value = next(
            (c["value"] for c in sample["conversations"]
             if c["from"] in ("assistant", "gpt")),
            "",
        )
        gt_action = parse_gt_action(assistant_value)
        if gt_action is None:
            print(f"[shard {args.shard_id}] idx={idx}: no GT action, skipping")
            continue

        # Image sizes.
        wh = get_image_size(sample)
        if wh is None:
            print(f"[shard {args.shard_id}] idx={idx}: no image, skipping")
            continue
        image_w, image_h = wh
        resized = get_resized_size(processor, sample) or (image_w, image_h)
        resized_w, resized_h = resized

        # Build messages and generate.
        try:
            messages = build_user_messages(sample)
            response = cooperative_generate(
                model=model,
                processor=processor,
                messages=messages,
                device=model_device,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        except Exception as e:
            print(f"[shard {args.shard_id}] idx={idx}: generate error: {e}")
            response = ""

        # Parse pred and compute credit.
        pred_action = parse_action_from_text(response)
        if pred_action is None:
            n_parse_fail += 1

        # Model outputs coordinates in original image space, not resized.
        # Use original (image_w, image_h) for both GT and pred normalization.
        credit = credit_for_sample(
            pred_action=pred_action,
            gt_action=gt_action,
            image_w=image_w,
            image_h=image_h,
            pred_image_w=image_w,
            pred_image_h=image_h,
        )

        record = {
            "sample_idx": idx,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "parse_ok": pred_action is not None,
            "image_size": [image_w, image_h],
            "resized_size": [resized_w, resized_h],
            **credit,
            "response_preview": response[:300],
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

        n_processed += 1
        if n_processed % 25 == 0:
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 1e-6)
            eta = (len(shard_indices) - n_processed) / max(rate, 1e-6)
            print(f"[shard {args.shard_id}] {n_processed}/{len(shard_indices)} "
                  f"parse_fail={n_parse_fail} rate={rate:.2f}/s ETA={eta/60:.1f}min")

    fout.close()
    print(f"[shard {args.shard_id}] Done. Output: {out_path} "
          f"(parse_fail={n_parse_fail}/{n_processed})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True,
                   help="Path to Qwen2.5-VL base model.")
    p.add_argument("--coop_checkpoint", required=True,
                   help="Cooperative LoRA checkpoint dir (with cooperative_config.json).")
    p.add_argument("--train_jsonl", required=True,
                   help="Single-turn training jsonl (e.g. ac_train_thought.jsonl).")
    p.add_argument("--out_dir", required=True,
                   help="Where credit_cache_shard{N}.jsonl will be written.")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 = all samples in shard")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--cuda_device", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing credit_cache_shard{N}.jsonl")
    args = p.parse_args()
    main(args)
