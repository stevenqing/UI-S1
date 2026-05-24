#!/usr/bin/env python3
"""Attention alignment analysis for coordinate predictions.

Loads model directly (not vLLM) to extract attention weights.
For correct vs wrong click predictions:
  1. Where does the model attend when generating coordinate tokens?
  2. Does attention center-of-mass align with predicted coordinate?
  3. Can alignment score distinguish correct from wrong predictions?

Also does crop-verify: crops region around predicted coordinate,
asks model "what element is here?", checks consistency with instruction.

Usage:
    python v15_gui_360/analyze_attention_alignment.py \
        --model_path checkpoints/gui360_balanced_full_sft_step272 \
        --bon_data v15_gui_360/outputs/preliminary_tests/bon_metrics/bon_raw_*.json \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --output_dir v15_gui_360/outputs/preliminary_tests/attention_analysis \
        --max_samples 100
"""

import argparse
import json
import glob
import os
import re
import sys
import time
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    SUPPORTED_ACTIONS, USER_PROMPT_TEMPLATE, _format_action_for_history,
)
from v13_gui_360.reward import compute_step_reward


def load_model_and_processor(model_path):
    """Load Qwen2.5-VL with eager attention for attention extraction."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading model from {model_path} with eager attention...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Model loaded. Device map: {set(str(p.device) for p in model.parameters())}")
    return model, processor


def find_image_token_positions(input_ids, processor):
    """Find the range of image tokens in the input sequence.

    Qwen2.5-VL places image tokens between <|vision_start|> and <|vision_end|>.
    """
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    ids = input_ids[0].tolist()
    start_idx = None
    end_idx = None
    for i, tid in enumerate(ids):
        if tid == vision_start_id:
            start_idx = i + 1  # tokens start AFTER <|vision_start|>
        if tid == vision_end_id:
            end_idx = i  # tokens end BEFORE <|vision_end|>
            break

    return start_idx, end_idx


def find_coordinate_tokens(input_ids, generated_ids, processor):
    """Find token positions of coordinate values in generated output.

    Looks for patterns like [520, 400] or "coordinate": [520, 400]
    Returns list of (token_idx_in_full_seq, is_x_coord).
    """
    # Decode generated tokens
    gen_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    prompt_len = input_ids.shape[1]

    # Find coordinate pattern in text
    coord_match = re.search(r'\[(\d+),\s*(\d+)\]', gen_text)
    if not coord_match:
        return [], None, None

    pred_x = int(coord_match.group(1))
    pred_y = int(coord_match.group(2))

    # Find token positions by searching for the coordinate string
    coord_str = coord_match.group(0)  # e.g., "[520, 400]"
    coord_start_char = coord_match.start()

    # Map character positions to token positions in generated text
    coord_token_indices = []
    char_count = 0
    gen_tokens = generated_ids[0][prompt_len:].tolist()

    for tok_idx, tok_id in enumerate(gen_tokens):
        tok_text = processor.tokenizer.decode([tok_id])
        tok_start = char_count
        tok_end = char_count + len(tok_text)

        # Check if this token overlaps with coordinate string
        if tok_start < coord_match.end() and tok_end > coord_match.start():
            # Check if it contains digits (actual coordinate values)
            if any(c.isdigit() for c in tok_text):
                coord_token_indices.append(prompt_len + tok_idx)

        char_count = tok_end

    return coord_token_indices, pred_x, pred_y


def compute_attention_center_of_mass(
    attentions, coord_token_indices, img_start, img_end, image_grid_thw, orig_w, orig_h
):
    """Compute attention-weighted center of mass over image tokens.

    Args:
        attentions: list of attention tensors [batch, heads, seq_len, seq_len]
        coord_token_indices: token positions of coordinate digits
        img_start, img_end: range of image tokens in sequence
        image_grid_thw: [t, h, w] grid dimensions
        orig_w, orig_h: original image dimensions

    Returns:
        (attn_x, attn_y): attention center of mass in original pixel coords
        attn_map: 2D attention heatmap over image grid
    """
    if not coord_token_indices or img_start is None:
        return None, None, None

    num_img_tokens = img_end - img_start
    h_grid = image_grid_thw[0][1].item() if torch.is_tensor(image_grid_thw[0][1]) else image_grid_thw[0][1]
    w_grid = image_grid_thw[0][2].item() if torch.is_tensor(image_grid_thw[0][2]) else image_grid_thw[0][2]

    # Average attention across last layers and all heads
    # For each coord token, get its attention to image tokens
    all_attn = []
    for attn in attentions:
        # attn: [batch, heads, seq_len, seq_len]
        attn_cpu = attn[0].float().cpu()  # [heads, seq_len, seq_len]
        for coord_idx in coord_token_indices:
            if coord_idx < attn_cpu.shape[1]:
                # Attention from coord_token to image tokens
                img_attn = attn_cpu[:, coord_idx, img_start:img_end]  # [heads, num_img_tokens]
                all_attn.append(img_attn.mean(dim=0))  # average over heads

    if not all_attn:
        return None, None, None

    # Average across all coord tokens and layers
    avg_attn = torch.stack(all_attn).mean(dim=0)  # [num_img_tokens]

    # Reshape to spatial grid
    if avg_attn.shape[0] != h_grid * w_grid:
        # Mismatch — try to handle temporal dimension
        t_grid = image_grid_thw[0][0].item() if torch.is_tensor(image_grid_thw[0][0]) else image_grid_thw[0][0]
        expected = t_grid * h_grid * w_grid
        if avg_attn.shape[0] == expected:
            # Average over temporal dimension
            avg_attn = avg_attn.view(t_grid, h_grid, w_grid).mean(dim=0).view(-1)
        else:
            # Can't map — truncate or pad
            min_len = min(avg_attn.shape[0], h_grid * w_grid)
            avg_attn = avg_attn[:min_len]
            if min_len < h_grid * w_grid:
                avg_attn = F.pad(avg_attn, (0, h_grid * w_grid - min_len))

    attn_2d = avg_attn[:h_grid * w_grid].view(h_grid, w_grid)

    # Normalize to probability distribution
    attn_2d = attn_2d / (attn_2d.sum() + 1e-8)

    # Compute center of mass in normalized coordinates
    y_coords = torch.arange(h_grid).float()
    x_coords = torch.arange(w_grid).float()

    attn_y_norm = (attn_2d.sum(dim=1) * y_coords).sum() / h_grid
    attn_x_norm = (attn_2d.sum(dim=0) * x_coords).sum() / w_grid

    # Convert to original pixel coordinates
    attn_x = attn_x_norm.item() * orig_w
    attn_y = attn_y_norm.item() * orig_h

    return attn_x, attn_y, attn_2d.numpy()


def analyze_sample(model, processor, screenshot_path, goal, history_text,
                   gt_action, image_w, image_h):
    """Analyze a single sample: generate, extract attention, compute alignment."""
    from qwen_vl_utils import process_vision_info

    # Build prompt
    prompt_text = USER_PROMPT_TEMPLATE.format(
        instruction=goal,
        history=history_text,
        actions=SUPPORTED_ACTIONS,
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{screenshot_path}"},
        {"type": "text", "text": prompt_text},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]

    # Step 1: Generate response
    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
        )

    gen_text = processor.tokenizer.decode(gen_output[0][prompt_len:], skip_special_tokens=True)

    # Parse predicted action
    from v13_gui_360.eval_gui360_template import parse_tool_call
    pred_action = parse_tool_call(gen_text)
    if pred_action is None:
        m = re.search(r'<action>\s*(\{.*?\})\s*</action>', gen_text, re.DOTALL)
        if m:
            try:
                pred_action = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # Compute reward
    if pred_action:
        fake_text = f"<action>{json.dumps(pred_action)}</action>"
    else:
        fake_text = gen_text
    reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
    is_correct = reward >= 0.5

    pred_coord = (pred_action or {}).get("coordinate")
    pred_type = (pred_action or {}).get("action", "")

    # Only do attention analysis for click actions with valid coordinates
    attn_result = None
    if pred_type == "click" and pred_coord and pred_coord[0] is not None:
        # Step 2: Forward pass with hook-based selective attention capture.
        # output_attentions=True stores ALL 28 layers' attention matrices
        # (~6.3GB for seq_len=2000), causing OOM.  Instead, register hooks
        # on every decoder layer: hooks on the last 4 layers save attention
        # to CPU; ALL hooks replace attention with None in the output tuple,
        # so the model's internal `all_self_attns` accumulates only Nones.
        try:
            # Qwen2.5-VL structure: model.model.language_model.layers
            decoder_layers = model.model.language_model.layers
            num_layers = len(decoder_layers)
            keep_last_n = 4
            target_set = set(range(num_layers - keep_last_n, num_layers))
            captured_attentions = {}

            def _make_hook(layer_idx):
                def _hook(module, inp, output):
                    # Debug: log output structure for first target layer
                    if layer_idx == num_layers - 1 and len(captured_attentions) == 0:
                        print(f"  [hook debug] layer {layer_idx}: output type={type(output)}, "
                              f"len={len(output) if isinstance(output, tuple) else 'N/A'}, "
                              f"types={[type(o).__name__ for o in output] if isinstance(output, tuple) else 'N/A'}")
                        for oi, o in enumerate(output if isinstance(output, tuple) else [output]):
                            if o is None:
                                print(f"    output[{oi}] = None")
                            elif hasattr(o, 'shape'):
                                print(f"    output[{oi}] = Tensor shape={o.shape}")
                            else:
                                print(f"    output[{oi}] = {type(o).__name__}")
                    if layer_idx in target_set and isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        captured_attentions[layer_idx] = output[1].detach().cpu()
                    # Drop attention from output → model won't accumulate it
                    if isinstance(output, tuple) and len(output) > 1:
                        return (output[0], None) + output[2:]
                    return output
                return _hook

            hooks = []
            for i, layer in enumerate(decoder_layers):
                hooks.append(layer.register_forward_hook(_make_hook(i)))

            with torch.no_grad():
                outputs = model(
                    input_ids=gen_output,
                    attention_mask=torch.ones_like(gen_output),
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    output_attentions=True,
                )

            for h in hooks:
                h.remove()
            del outputs
            torch.cuda.empty_cache()

            # Reconstruct attention list (last 4 layers, on CPU)
            if not captured_attentions:
                print(f"  [attention] No attention captured (0/{keep_last_n} target layers)")
            else:
                print(f"  [attention] Captured {len(captured_attentions)}/{keep_last_n} layers")
            attentions = [captured_attentions[i] for i in sorted(captured_attentions)]

            # Find image token range
            img_start, img_end = find_image_token_positions(gen_output, processor)

            # Find coordinate token positions
            coord_indices, px, py = find_coordinate_tokens(
                inputs["input_ids"], gen_output, processor
            )

            if img_start is not None and coord_indices:
                attn_x, attn_y, attn_map = compute_attention_center_of_mass(
                    attentions, coord_indices, img_start, img_end,
                    inputs["image_grid_thw"], image_w, image_h
                )

                if attn_x is not None:
                    # Compute alignment: distance between attention CoM and predicted coord
                    align_dist = ((attn_x - pred_coord[0])**2 + (attn_y - pred_coord[1])**2) ** 0.5
                    diag = (image_w**2 + image_h**2) ** 0.5

                    attn_result = {
                        "attn_center_x": float(attn_x),
                        "attn_center_y": float(attn_y),
                        "pred_x": float(pred_coord[0]),
                        "pred_y": float(pred_coord[1]),
                        "alignment_dist": float(align_dist),
                        "alignment_ratio": float(align_dist / diag),
                    }

            del attentions, captured_attentions
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Attention extraction error: {e}")
            import traceback; traceback.print_exc()

    return {
        "is_correct": is_correct,
        "reward": float(reward),
        "pred_type": pred_type,
        "gt_type": gt_action.get("action", ""),
        "pred_coord": pred_coord,
        "gt_coord": gt_action.get("coordinate"),
        "gen_text": gen_text[:300],
        "attention": attn_result,
    }


def run_crop_verify(model, processor, screenshot_path, pred_coord, goal, image_w, image_h):
    """Crop around predicted coordinate and ask model to identify the element."""
    from qwen_vl_utils import process_vision_info

    if not pred_coord or pred_coord[0] is None:
        return None

    # Crop 200x200 region around predicted coordinate
    img = Image.open(screenshot_path)
    cx, cy = int(pred_coord[0]), int(pred_coord[1])
    crop_size = 100  # half-width
    left = max(0, cx - crop_size)
    top = max(0, cy - crop_size)
    right = min(img.width, cx + crop_size)
    bottom = min(img.height, cy + crop_size)
    crop = img.crop((left, top, right, bottom))

    # Save crop temporarily
    crop_buf = BytesIO()
    crop.save(crop_buf, format="PNG")
    crop_buf.seek(0)

    # Step 1: Ask model what element is in the crop
    identify_prompt = (
        "What UI element is shown in this cropped image? "
        "Describe it in one short sentence (e.g., 'a search text box', "
        "'the Settings icon', 'a blue Submit button')."
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop},
        {"type": "text", "text": identify_prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, temperature=0.0, do_sample=False)
    element_desc = processor.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Step 2: Ask if this element matches the instruction
    verify_prompt = (
        f'The user instruction is: "{goal}"\n'
        f'The UI element at the predicted click location is: "{element_desc.strip()}"\n\n'
        f'Is this the correct element to interact with for this instruction? '
        f'Answer only YES or NO.'
    )

    messages2 = [{"role": "user", "content": [{"type": "text", "text": verify_prompt}]}]
    text2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    inputs2 = processor(text=[text2], return_tensors="pt")
    inputs2 = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs2.items()}

    with torch.no_grad():
        gen2 = model.generate(**inputs2, max_new_tokens=16, temperature=0.0, do_sample=False)
    verify_answer = processor.tokenizer.decode(gen2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True)

    is_yes = verify_answer.strip().upper().startswith("YES")

    return {
        "element_description": element_desc.strip()[:200],
        "verify_answer": verify_answer.strip()[:50],
        "verified_correct": is_yes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--bon_data", default=None,
                        help="Path to bon_raw_*.json to identify partial steps")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--do_crop_verify", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    episodes = {}
    with open(args.test_data) as f:
        for line in f:
            ep = json.loads(line)
            episodes[str(ep["episode_id"])] = ep

    # If bon_data provided, focus on partial steps (most informative)
    target_samples = []
    if args.bon_data:
        bon_files = sorted(glob.glob(args.bon_data))
        if bon_files:
            bon = json.load(open(bon_files[-1]))
            for eid, ep in bon.items():
                if eid not in episodes:
                    continue
                for step in ep["steps"]:
                    samples = step["samples"]
                    n_correct = sum(1 for s in samples if s.get("is_correct"))
                    gt_type = step.get("gt_type", "")
                    # Focus on partial click steps (where verifier matters most)
                    if 1 <= n_correct <= 4 and gt_type == "click":
                        target_samples.append((eid, step["step_idx"]))
            print(f"Found {len(target_samples)} partial click steps from BoN data")
    else:
        # Use first N episodes, all steps
        for eid, ep in list(episodes.items())[:50]:
            for i, step in enumerate(ep["steps"]):
                if step["action"].get("action") == "click":
                    target_samples.append((eid, i))

    # Limit
    if len(target_samples) > args.max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(target_samples), args.max_samples, replace=False)
        target_samples = [target_samples[i] for i in sorted(indices)]
    print(f"Will analyze {len(target_samples)} samples")

    # Load model
    model, processor = load_model_and_processor(args.model_path)

    # Run analysis
    results = []
    correct_aligns = []
    wrong_aligns = []
    crop_tp, crop_fp, crop_tn, crop_fn = 0, 0, 0, 0

    for idx, (eid, step_idx) in enumerate(tqdm(target_samples, desc="Analyzing")):
        ep = episodes[eid]
        step = ep["steps"][step_idx]

        # Build GT history up to this step
        history = []
        for j in range(step_idx):
            history.append(_format_action_for_history(ep["steps"][j]["action"], j + 1))
        history_text = "\n".join(history) if history else "None"

        gt_action = step["action"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        # Attention analysis
        result = analyze_sample(
            model, processor, step["screenshot"], ep["goal"], history_text,
            gt_action, image_w, image_h
        )
        result["episode_id"] = eid
        result["step_idx"] = step_idx

        if result["attention"] is not None:
            ad = result["attention"]["alignment_dist"]
            if result["is_correct"]:
                correct_aligns.append(ad)
            else:
                wrong_aligns.append(ad)

        # Crop verification
        if args.do_crop_verify and result["pred_type"] == "click" and result["pred_coord"]:
            crop_result = run_crop_verify(
                model, processor, step["screenshot"],
                result["pred_coord"], ep["goal"], image_w, image_h
            )
            result["crop_verify"] = crop_result

            if crop_result:
                actually_correct = result["is_correct"]
                verified_correct = crop_result["verified_correct"]
                if verified_correct and actually_correct:
                    crop_tp += 1
                elif verified_correct and not actually_correct:
                    crop_fp += 1
                elif not verified_correct and not actually_correct:
                    crop_tn += 1
                else:
                    crop_fn += 1

        results.append(result)

        # Print periodic updates
        if (idx + 1) % 10 == 0:
            if correct_aligns and wrong_aligns:
                print(f"  [{idx+1}] Attn align — correct: {np.mean(correct_aligns):.1f}px, "
                      f"wrong: {np.mean(wrong_aligns):.1f}px")
            if crop_tp + crop_fp + crop_tn + crop_fn > 0:
                total_cv = crop_tp + crop_fp + crop_tn + crop_fn
                cv_acc = (crop_tp + crop_tn) / total_cv
                print(f"  [{idx+1}] Crop verify — acc={cv_acc:.1%} "
                      f"TP={crop_tp} FP={crop_fp} TN={crop_tn} FN={crop_fn}")

    # ══ Summary ══
    print(f"\n{'='*70}")
    print("Attention Alignment Analysis")
    print(f"{'='*70}")

    if correct_aligns and wrong_aligns:
        ca = np.array(correct_aligns)
        wa = np.array(wrong_aligns)
        print(f"\nAttention-coordinate distance (pixels):")
        print(f"  Correct predictions: mean={ca.mean():.1f}, median={np.median(ca):.1f}")
        print(f"  Wrong predictions:   mean={wa.mean():.1f}, median={np.median(wa):.1f}")
        print(f"  Gap:                 {wa.mean() - ca.mean():.1f}px")

        # Cohen's d
        pooled_std = ((ca.std()**2 + wa.std()**2) / 2) ** 0.5
        if pooled_std > 0:
            d = (wa.mean() - ca.mean()) / pooled_std
            print(f"  Cohen's d:           {d:.3f}")

        # Selection accuracy: pick sample with smaller alignment distance
        print(f"\n  Correct with attn_dist < 100px: {(ca < 100).sum()}/{len(ca)} = {(ca < 100).mean()*100:.1f}%")
        print(f"  Wrong with attn_dist < 100px:   {(wa < 100).sum()}/{len(wa)} = {(wa < 100).mean()*100:.1f}%")
        print(f"  Correct with attn_dist < 200px: {(ca < 200).sum()}/{len(ca)} = {(ca < 200).mean()*100:.1f}%")
        print(f"  Wrong with attn_dist < 200px:   {(wa < 200).sum()}/{len(wa)} = {(wa < 200).mean()*100:.1f}%")

    print(f"\n{'='*70}")
    print("Crop-and-Verify Analysis")
    print(f"{'='*70}")

    total_cv = crop_tp + crop_fp + crop_tn + crop_fn
    if total_cv > 0:
        print(f"\nConfusion matrix:")
        print(f"  True Positive  (verify=YES, actually correct):  {crop_tp}")
        print(f"  False Positive (verify=YES, actually wrong):    {crop_fp}")
        print(f"  True Negative  (verify=NO,  actually wrong):    {crop_tn}")
        print(f"  False Negative (verify=NO,  actually correct):  {crop_fn}")
        print(f"\n  Accuracy:    {(crop_tp + crop_tn) / total_cv * 100:.1f}%")
        if crop_tp + crop_fp > 0:
            print(f"  Precision:   {crop_tp / (crop_tp + crop_fp) * 100:.1f}%")
        if crop_tp + crop_fn > 0:
            print(f"  Recall:      {crop_tp / (crop_tp + crop_fn) * 100:.1f}%")
        if crop_tn + crop_fp > 0:
            print(f"  Rejection rate: {(crop_tn + crop_fn) / total_cv * 100:.1f}%")
            print(f"  (vs self-verify: 0.04% rejection rate)")

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Clean results for JSON serialization
    clean_results = []
    for r in results:
        cr = dict(r)
        if cr.get("attention") and "attn_map" in cr["attention"]:
            del cr["attention"]["attn_map"]  # numpy array
        clean_results.append(cr)

    with open(os.path.join(args.output_dir, f"attention_analysis_{ts}.json"), "w") as f:
        json.dump(clean_results, f, indent=2)

    summary = {
        "num_samples": len(results),
        "attention": {
            "correct_mean_dist": float(np.mean(correct_aligns)) if correct_aligns else None,
            "wrong_mean_dist": float(np.mean(wrong_aligns)) if wrong_aligns else None,
            "num_correct": len(correct_aligns),
            "num_wrong": len(wrong_aligns),
        },
        "crop_verify": {
            "tp": crop_tp, "fp": crop_fp, "tn": crop_tn, "fn": crop_fn,
            "accuracy": (crop_tp + crop_tn) / total_cv if total_cv > 0 else 0,
        },
    }
    with open(os.path.join(args.output_dir, f"attention_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
