#!/usr/bin/env python3
"""
Exp D: Micro Grounding SFT

Train LoRA for 1 epoch on 200 grounding samples, then re-run Probe C
to see if cross-modal binding improves.

Usage:
  python train_micro_grounding.py \
      --base_model /path/to/model \
      --output_dir /path/to/output \
      --then_probe  # optionally run Probe C after training
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


def parse_tool_call(text):
    """Extract tool call JSON from <tool_call>...</tool_call> tags."""
    if not text:
        return None
    m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


class GroundingDataset(Dataset):
    def __init__(self, data_path, processor):
        with open(data_path) as f:
            self.data = [json.loads(line) for line in f]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        screenshot = item["screenshot"]
        goal = item["goal"]
        thought_first = item["thought_first"]
        coord = item["coord"]

        image = Image.open(screenshot).convert("RGB")

        user_text = (
            f"You are a GUI grounding assistant. Given a screenshot and a task instruction, "
            f"locate the UI element that should be interacted with next.\n\n"
            f"Task: {goal}\n\n"
            f"Context: {thought_first}\n\n"
            f"Output the coordinates of the target element."
        )

        assistant_text = (
            f"The target element is located at coordinates ({coord[0]}, {coord[1]}) in the screenshot."
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": assistant_text},
            ]},
        ]

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False)

        inputs = self.processor(
            text=[prompt], images=[image],
            return_tensors="pt", padding=True)

        # Remove batch dim
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs


def collate_fn(batch):
    """Simple collate — just use batch size 1."""
    return batch[0]


def train(model, dataset, args):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    model.train()
    total_loss = 0
    n_steps = 0

    print(f"Training {len(dataset)} samples, lr={args.lr}")

    for epoch in range(args.epochs):
        indices = np.random.permutation(len(dataset))
        epoch_loss = 0

        for i, idx in enumerate(indices):
            try:
                inputs = dataset[idx]
            except Exception as e:
                print(f"  Skip sample {idx}: {e}")
                continue

            inputs = {k: v.unsqueeze(0).to(model.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            # Mask everything before the assistant response
            # Find the assistant marker token
            ids_list = input_ids.squeeze().tolist()
            # Find last occurrence of "assistant" tokens — look for the pattern
            # In Qwen2.5-VL, assistant turn starts with <|im_start|>assistant
            IM_START = 151644
            asst_positions = []
            for j in range(len(ids_list) - 1):
                if ids_list[j] == IM_START:
                    asst_positions.append(j)

            if len(asst_positions) >= 2:
                # Last im_start is the assistant turn
                mask_end = asst_positions[-1] + 2  # skip <|im_start|>assistant\n
            else:
                mask_end = len(ids_list) // 2  # fallback

            labels[0, :mask_end] = -100

            try:
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                total_loss += loss.item()
                n_steps += 1

            except Exception as e:
                print(f"  Step {i} error: {e}")
                optimizer.zero_grad()
                continue

            if (i + 1) % 20 == 0:
                avg_loss = epoch_loss / (i + 1)
                print(f"  Epoch {epoch + 1} [{i + 1}/{len(dataset)}] loss={avg_loss:.4f}")

            del outputs, inputs
            torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / max(len(dataset), 1)
        print(f"Epoch {epoch + 1} done. Avg loss = {avg_epoch_loss:.4f}")

    avg_total = total_loss / max(n_steps, 1)
    print(f"\nTraining complete. Total steps={n_steps}, avg loss={avg_total:.4f}")


def _identify_task_tokens(input_ids, tokenizer):
    ids = input_ids.squeeze().tolist()
    cum_len = 0
    token_char_starts = []
    for tid in ids:
        token_char_starts.append(cum_len)
        cum_len += len(tokenizer.decode([tid]))
    full_text = tokenizer.decode(ids)

    def find_pos(marker):
        pos = full_text.rfind(marker)
        if pos == -1:
            return None
        for i in range(len(token_char_starts) - 1, -1, -1):
            if token_char_starts[i] <= pos:
                return i
        return None

    instr_pos = find_pos("instruction is:\n")
    hist_pos = find_pos("history of actions are:\n")
    if instr_pos is None:
        return []
    task_end = hist_pos if hist_pos is not None else len(ids)
    if hist_pos is not None:
        task_end = max(0, hist_pos - 2)
    task_start = max(0, instr_pos - 2)
    return list(range(task_start, task_end))


def run_probe_c(model, processor, tokenizer, parquet_path, args):
    """Re-run Probe C after grounding training."""
    import pandas as pd

    VISION_START_ID = 151652
    VISION_END_ID = 151653
    SPATIAL_MERGE_SIZE = 2
    PATCH_SIZE = 14
    TOKEN_PIXEL_SIZE = SPATIAL_MERGE_SIZE * PATCH_SIZE

    df = pd.read_parquet(parquet_path)
    np.random.seed(42)
    if args.probe_n < len(df):
        indices = np.random.choice(len(df), args.probe_n, replace=False)
        df = df.iloc[indices].reset_index(drop=True)

    print(f"\nRunning Probe C on {len(df)} samples...")
    model.eval()

    probe_layers = [0, 14, 27]
    sims_data = {li: {"target": [], "nontarget": []} for li in probe_layers}

    n_done = 0
    for idx in range(len(df)):
        row = df.iloc[idx]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        gt_action = parse_tool_call(messages[1]["content"])
        if gt_action is None:
            continue
        gt_bbox = gt_action.get("bbox")
        if not gt_bbox or any(gt_bbox.get(k) is None for k in ["left", "top", "right", "bottom"]):
            continue

        text_content = ""
        image_path = None
        for item in messages[0]["content"]:
            if isinstance(item, dict):
                if "text" in item:
                    text_content = item["text"]
                if "image" in item:
                    image_path = item["image"]

        if not image_path:
            continue
        full_path = os.path.join(args.image_base, image_path)
        if not os.path.exists(full_path):
            continue

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception:
            continue

        orig_w, orig_h = image.size

        prompt_messages = [{"role": "user", "content": [
            {"type": "text", "text": text_content},
            {"type": "image", "image": full_path},
        ]}]
        prompt_text = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False)

        try:
            inputs = processor(
                text=[prompt_text], images=[image],
                return_tensors="pt", padding=True)
        except Exception:
            continue

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None or len(image_grid_thw) == 0:
            continue
        grid_thw = image_grid_thw[0].tolist()

        token_h = grid_thw[1] // SPATIAL_MERGE_SIZE
        token_w = grid_thw[2] // SPATIAL_MERGE_SIZE
        n_image_tokens = token_h * token_w

        resized_w = grid_thw[2] * PATCH_SIZE
        resized_h = grid_thw[1] * PATCH_SIZE
        scale_w = resized_w / orig_w
        scale_h = resized_h / orig_h

        bl = gt_bbox["left"] * scale_w
        bt = gt_bbox["top"] * scale_h
        br = gt_bbox["right"] * scale_w
        bb = gt_bbox["bottom"] * scale_h

        ids_list = input_ids.squeeze().tolist()
        img_start = img_end = None
        for i, t in enumerate(ids_list):
            if t == VISION_START_ID and img_start is None:
                img_start = i
            if t == VISION_END_ID:
                img_end = i

        if img_start is None or img_end is None:
            continue

        img_token_start = img_start + 1
        img_token_end = img_end
        if img_token_end - img_token_start != n_image_tokens:
            continue

        # Target tokens
        positions = []
        for row_i in range(token_h):
            for col_i in range(token_w):
                y1 = row_i * TOKEN_PIXEL_SIZE
                x1 = col_i * TOKEN_PIXEL_SIZE
                positions.append((x1, y1, x1 + TOKEN_PIXEL_SIZE, y1 + TOKEN_PIXEL_SIZE))

        target_set = set()
        for i, (x1, y1, x2, y2) in enumerate(positions):
            if x2 > bl and x1 < br and y2 > bt and y1 < bb:
                target_set.add(i)

        if not target_set:
            continue

        # Task text tokens
        task_indices = _identify_task_tokens(input_ids, tokenizer)
        if not task_indices:
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        for li in probe_layers:
            if li + 1 >= len(outputs.hidden_states):
                continue
            hs = outputs.hidden_states[li + 1][0].float()
            img_feats = hs[img_token_start:img_token_end]
            task_mean = hs[task_indices].mean(dim=0)

            target_list = list(target_set)
            nontarget_list = [i for i in range(n_image_tokens) if i not in target_set]

            target_feats = img_feats[target_list]
            nontarget_feats = img_feats[nontarget_list]

            t_mean = target_feats.mean(dim=0)
            nt_mean = nontarget_feats.mean(dim=0)

            t_sim = F.cosine_similarity(t_mean.unsqueeze(0), task_mean.unsqueeze(0)).item()
            nt_sim = F.cosine_similarity(nt_mean.unsqueeze(0), task_mean.unsqueeze(0)).item()

            sims_data[li]["target"].append(t_sim)
            sims_data[li]["nontarget"].append(nt_sim)

        del outputs
        torch.cuda.empty_cache()

        n_done += 1
        if n_done % 20 == 0:
            print(f"  Probe C: {n_done} done")

    print(f"\n── Probe C Results (after grounding training) ──")
    print(f"{'Layer':>6} | {'target-task':>12} | {'nontarget-task':>14} | {'gap':>8}")
    print("-" * 50)

    results = {}
    for li in probe_layers:
        if not sims_data[li]["target"]:
            continue
        t_mean = np.mean(sims_data[li]["target"])
        nt_mean = np.mean(sims_data[li]["nontarget"])
        gap = t_mean - nt_mean
        print(f"{li:>6} | {t_mean:>12.4f} | {nt_mean:>14.4f} | {gap:>+8.4f}")
        results[li] = {"target": round(t_mean, 4), "nontarget": round(nt_mean, 4), "gap": round(gap, 4)}

    with open(os.path.join(args.output_dir, "probe_c_after_grounding.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Micro Grounding SFT (Exp D)")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--data_path",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/gui360_grounding_200_simple.jsonl")
    parser.add_argument("--parquet_file",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet")
    parser.add_argument("--image_base", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--then_probe", action="store_true",
                        help="Run Probe C after training")
    parser.add_argument("--probe_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print(f"Base model: {args.base_model}")
    print(f"Data: {args.data_path}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print()

    # Load model
    print("Loading model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Add LoRA
    print("Adding LoRA...", flush=True)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("Loading grounding dataset...", flush=True)
    dataset = GroundingDataset(args.data_path, processor)
    print(f"Dataset: {len(dataset)} samples")

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    t0 = time.time()
    train(model, dataset, args)
    print(f"Training time: {time.time() - t0:.1f}s")

    # Save
    save_path = os.path.join(args.output_dir, "grounding_lora")
    model.save_pretrained(save_path)
    print(f"LoRA saved to {save_path}")

    # Optionally run Probe C
    if args.then_probe:
        print("\n" + "=" * 60)
        print("RUNNING PROBE C AFTER GROUNDING TRAINING")
        print("=" * 60)
        run_probe_c(model, processor, tokenizer, args.parquet_file, args)


if __name__ == "__main__":
    main()
