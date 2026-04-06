"""
Phase 0: Inference-time Attention Steering on SFT v2.

Tests whether representational fragmentation can be bridged by steering
attention toward binding-relevant image tokens, without any parameter changes.

Modes:
  baseline       — No intervention (SFT v2 as-is)
  binding_boost  — (I1) Boost attention scores at binding-relevant image tokens
  key_amplify    — (I2) Amplify k_proj output at binding-relevant positions
  random_boost   — Control: boost random image tokens (same count as binding)

Approach:
  1. First pass: extract hidden states from layer L_bind (default 24)
  2. Compute cosine similarity between each image token and task text centroid
  3. Top-P% image tokens = "binding-relevant"
  4. Register attention/key hooks on target layers
  5. Generate with hooks active
  6. Evaluate predicted action vs ground truth

Usage:
  python attention_steering_eval.py \
      --model_path .../gui360_full_sft_v2 \
      --mode binding_boost \
      --alpha 2.0 \
      --n_samples 500
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helpers (from eval_multiview_gui360.py)
# ═══════════════════════════════════════════════════════════════════════

def parse_tool_call(text):
    try:
        match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None


def evaluate_action(pred_action, gt_action, threshold=50, use_bbox=True):
    result = {"function_match": False, "args_match": False,
              "full_match": False, "bbox_match": None}
    if pred_action is None or gt_action is None:
        return result

    pred_func = pred_action.get("function", "")
    gt_func = gt_action.get("function", "")
    result["function_match"] = (pred_func == gt_func)
    if not result["function_match"]:
        return result

    pred_args = pred_action.get("args", {})
    gt_args = gt_action.get("args", {})
    gt_bbox = gt_action.get("bbox", None)

    if pred_func == "click":
        pred_coord = pred_args.get("coordinate", [])
        gt_coord = gt_args.get("coordinate", [])
        if len(pred_coord) == 2:
            if use_bbox and gt_bbox:
                left, top = gt_bbox.get("left"), gt_bbox.get("top")
                right, bottom = gt_bbox.get("right"), gt_bbox.get("bottom")
                if all(v is not None for v in [left, top, right, bottom]):
                    result["bbox_match"] = (left <= pred_coord[0] <= right and
                                            top <= pred_coord[1] <= bottom)
                    result["args_match"] = result["bbox_match"]
            if not result["args_match"] and len(gt_coord) == 2:
                dist = ((pred_coord[0] - gt_coord[0])**2 +
                        (pred_coord[1] - gt_coord[1])**2)**0.5
                result["args_match"] = dist < threshold
    elif pred_func == "type":
        pred_text = pred_args.get("text", "").lower().strip()
        gt_text = gt_args.get("text", "").lower().strip()
        result["args_match"] = (pred_text == gt_text)
    elif pred_func in ["wheel_mouse_input", "scroll"]:
        result["args_match"] = (pred_args.get("direction", "") ==
                                gt_args.get("direction", ""))
    elif pred_func == "summary":
        result["args_match"] = True
    else:
        result["args_match"] = (pred_args == gt_args)

    result["full_match"] = result["function_match"] and result["args_match"]
    return result


# ═══════════════════════════════════════════════════════════════════════
# Token identification
# ═══════════════════════════════════════════════════════════════════════

def find_image_token_range(input_ids):
    """Return (start, end) indices of image tokens (exclusive of markers)."""
    ids = input_ids.squeeze().tolist()
    img_start = img_end = None
    for i, t in enumerate(ids):
        if t == VISION_START_ID and img_start is None:
            img_start = i
        if t == VISION_END_ID:
            img_end = i
    if img_start is not None and img_end is not None:
        return img_start + 1, img_end  # exclusive range
    return None, None


def find_task_token_range(input_ids, tokenizer):
    """Find task instruction tokens (between 'instruction is:' and 'history of actions')."""
    text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=False)

    instr_marker = "instruction is:\n"
    hist_marker = "history of actions are:\n"

    instr_pos = text.find(instr_marker)
    hist_pos = text.find(hist_marker)

    if instr_pos < 0:
        return None, None

    # Map character positions to token positions
    ids = input_ids.squeeze().tolist()
    cumlen = 0
    char_to_token = {}
    for i, tid in enumerate(ids):
        tok_text = tokenizer.decode([tid])
        for c in range(len(tok_text)):
            char_to_token[cumlen + c] = i
        cumlen += len(tok_text)

    start_token = char_to_token.get(instr_pos, None)
    if hist_pos > 0:
        end_token = char_to_token.get(hist_pos, None)
    else:
        img_start, _ = find_image_token_range(input_ids)
        end_token = img_start

    return start_token, end_token


# ═══════════════════════════════════════════════════════════════════════
# Binding computation
# ═══════════════════════════════════════════════════════════════════════

def compute_binding_tokens(hidden_states, input_ids, tokenizer,
                           bind_layer=24, top_p=0.2):
    """
    Compute binding-relevant image tokens using cosine similarity
    between image token representations and task text centroid.

    Args:
        hidden_states: tuple of (B, seq_len, hidden_dim) from output_hidden_states=True
        input_ids: (1, seq_len)
        tokenizer: tokenizer
        bind_layer: which layer to extract features from
        top_p: fraction of image tokens to select as binding-relevant

    Returns:
        binding_indices: list of token indices (absolute positions in sequence)
        all_sims: cosine similarities for all image tokens
        task_centroid: task text centroid vector
    """
    # hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i
    hs = hidden_states[bind_layer + 1][0]  # (seq_len, hidden_dim)

    # Image tokens
    img_start, img_end = find_image_token_range(input_ids)
    if img_start is None:
        return [], None, None

    img_feats = hs[img_start:img_end]  # (n_img, hidden_dim)
    n_img = img_feats.shape[0]

    # Task text tokens
    task_start, task_end = find_task_token_range(input_ids, tokenizer)
    if task_start is None or task_end is None or task_start >= task_end:
        # Fallback: use all non-image tokens before image
        task_start, task_end = 0, img_start

    task_feats = hs[task_start:task_end]  # (n_task, hidden_dim)
    if task_feats.shape[0] == 0:
        return [], None, None

    task_centroid = task_feats.mean(dim=0)  # (hidden_dim,)

    # Cosine similarity
    sims = F.cosine_similarity(
        img_feats, task_centroid.unsqueeze(0), dim=-1)  # (n_img,)

    # Top P% as binding-relevant
    k = max(1, int(n_img * top_p))
    topk_vals, topk_local = sims.topk(k)

    # Convert to absolute indices
    binding_indices = [img_start + i.item() for i in topk_local]

    return binding_indices, sims.cpu(), task_centroid


# ═══════════════════════════════════════════════════════════════════════
# Hooks
# ═══════════════════════════════════════════════════════════════════════

class BindingAttentionBiasHook:
    """Pre-forward hook that boosts attention to binding-relevant tokens."""

    def __init__(self, binding_indices, alpha=2.0):
        self.binding_indices = binding_indices
        self.alpha = alpha

    def __call__(self, module, args, kwargs):
        attn_mask = kwargs.get("attention_mask")
        if attn_mask is not None and attn_mask.dim() == 4:
            mask = attn_mask.clone()
            key_len = mask.shape[-1]
            valid_idx = [i for i in self.binding_indices if i < key_len]
            if valid_idx:
                idx_t = torch.tensor(valid_idx, device=mask.device)
                mask[:, :, :, idx_t] += self.alpha
            kwargs["attention_mask"] = mask
        return args, kwargs


class KeyAmplifyHook:
    """Forward hook that amplifies k_proj output at binding-relevant positions."""

    def __init__(self, binding_indices, beta=0.5):
        self.binding_indices = binding_indices
        self.beta = beta
        self._prefill_done = False

    def __call__(self, module, input, output):
        # Only modify during prefill (seq_len > 1), not decode steps
        if output.shape[1] > 1 and not self._prefill_done:
            for idx in self.binding_indices:
                if idx < output.shape[1]:
                    output[0, idx] *= (1.0 + self.beta)
            self._prefill_done = True
        return output


def apply_binding_boost(model, binding_indices, alpha, target_layers):
    """Register attention bias hooks on target layers."""
    handles = []
    # Qwen2.5-VL: ForConditionalGeneration -> model -> language_model -> layers
    lang_model = model.model.language_model
    for layer_idx, layer in enumerate(lang_model.layers):
        if layer_idx in target_layers:
            hook = BindingAttentionBiasHook(binding_indices, alpha)
            h = layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
            handles.append(h)
    return handles


def apply_key_amplify(model, binding_indices, beta, target_layers):
    """Register k_proj amplification hooks on target layers."""
    handles = []
    lang_model = model.model.language_model
    for layer_idx, layer in enumerate(lang_model.layers):
        if layer_idx in target_layers:
            hook = KeyAmplifyHook(binding_indices, beta)
            h = layer.self_attn.k_proj.register_forward_hook(hook)
            handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ═══════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════

def process_sample(model, processor, tokenizer, row, args, idx):
    """Process one sample with the specified intervention mode."""
    messages = row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)

    user_msg = messages[0]
    gt_response = messages[1]["content"]
    gt_action = parse_tool_call(gt_response)

    # Extract text and image
    text_content = ""
    image_path = None
    for item in user_msg["content"]:
        if isinstance(item, dict):
            if "text" in item:
                text_content = item["text"]
            if "image" in item:
                image_path = item["image"]

    if image_path is None or gt_action is None:
        return None

    full_path = os.path.join(args.image_base, image_path)
    if not os.path.exists(full_path):
        return None

    try:
        image = Image.open(full_path).convert("RGB")
    except Exception:
        return None

    # Prepare input
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
        return None

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # ── Step 1: First pass to compute binding tokens ──
    binding_indices = []
    n_img_tokens = 0
    binding_sims = None

    if args.mode != "baseline":
        with torch.no_grad():
            first_outputs = model(
                **inputs, output_hidden_states=True, return_dict=True)

        hidden_states = first_outputs.hidden_states
        binding_indices, binding_sims, task_centroid = compute_binding_tokens(
            hidden_states, input_ids, tokenizer,
            bind_layer=args.bind_layer, top_p=args.top_p)

        img_start, img_end = find_image_token_range(input_ids)
        n_img_tokens = img_end - img_start if img_start is not None else 0

        # For random control: replace binding indices with random image tokens
        if args.mode == "random_boost" and img_start is not None:
            all_img_indices = list(range(img_start, img_end))
            k = len(binding_indices)
            rng = np.random.RandomState(idx)
            random_indices = rng.choice(all_img_indices, size=min(k, len(all_img_indices)),
                                        replace=False)
            binding_indices = sorted(random_indices.tolist())

        del first_outputs, hidden_states
        torch.cuda.empty_cache()

    # ── Step 2: Register hooks ──
    handles = []
    target_layers = set(range(args.layer_start, args.layer_end + 1))

    if args.mode in ["binding_boost", "random_boost"] and binding_indices:
        handles = apply_binding_boost(model, binding_indices, args.alpha, target_layers)
    elif args.mode == "key_amplify" and binding_indices:
        handles = apply_key_amplify(model, binding_indices, args.beta, target_layers)

    # ── Step 3: Generate ──
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    remove_hooks(handles)

    resp_ids = gen_ids[0, seq_len:]
    response = tokenizer.decode(resp_ids, skip_special_tokens=True)
    pred_action = parse_tool_call(response)
    eval_result = evaluate_action(pred_action, gt_action)

    del gen_ids
    torch.cuda.empty_cache()

    return {
        "sample_idx": int(idx),
        "correct": eval_result["full_match"],
        "function_match": eval_result["function_match"],
        "args_match": eval_result["args_match"],
        "bbox_match": eval_result.get("bbox_match"),
        "pred_function": pred_action.get("function", "") if pred_action else "",
        "gt_function": gt_action.get("function", "") if gt_action else "",
        "n_binding_tokens": len(binding_indices),
        "n_img_tokens": n_img_tokens,
        "pred_response": response[:200],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Inference-time Attention Steering")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_file",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/"
                                "train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet")
    parser.add_argument("--image_base",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", required=True,
                        choices=["baseline", "binding_boost", "key_amplify", "random_boost"],
                        help="Intervention mode")

    # Binding params
    parser.add_argument("--bind_layer", type=int, default=24,
                        help="Layer to extract binding features from")
    parser.add_argument("--top_p", type=float, default=0.2,
                        help="Fraction of image tokens to select as binding-relevant")

    # Intervention params
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Attention bias strength (for binding_boost/random_boost)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Key amplification factor (for key_amplify)")
    parser.add_argument("--layer_start", type=int, default=19,
                        help="First layer to apply intervention")
    parser.add_argument("--layer_end", type=int, default=27,
                        help="Last layer to apply intervention")

    # Eval params
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Phase 0: Attention Steering Evaluation")
    print(f"=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_path}")
    print(f"Bind layer: L{args.bind_layer}, top_p: {args.top_p}")
    if args.mode in ["binding_boost", "random_boost"]:
        print(f"Alpha: {args.alpha}, layers: L{args.layer_start}-L{args.layer_end}")
    elif args.mode == "key_amplify":
        print(f"Beta: {args.beta}, layers: L{args.layer_start}-L{args.layer_end}")
    print(f"Samples: {args.n_samples}")
    print()

    # Load model
    print("Loading model (bfloat16, eager attention)...", flush=True)
    t0 = time.time()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    print("Processor loaded", flush=True)

    # Load data
    print(f"Loading data: {args.parquet_file}", flush=True)
    df = pd.read_parquet(args.parquet_file)
    np.random.seed(args.seed)
    if 0 < args.n_samples < len(df):
        indices = np.random.choice(len(df), args.n_samples, replace=False)
        df = df.iloc[sorted(indices)].reset_index(drop=True)
    print(f"Processing {len(df)} samples\n")

    # Run evaluation
    results = []
    result_path = os.path.join(args.output_dir, f"{args.mode}_results.jsonl")

    for idx in range(len(df)):
        t_start = time.time()
        row = df.iloc[idx]

        result = process_sample(model, processor, tokenizer, row, args, idx)
        if result is None:
            continue

        results.append(result)

        # Append to file
        with open(result_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        elapsed = time.time() - t_start
        n_done = len(results)

        if n_done % 10 == 0:
            n_correct = sum(1 for r in results if r["correct"])
            n_fn = sum(1 for r in results if r["function_match"])
            print(f"[{n_done}/{len(df)}] {elapsed:.1f}s/sample | "
                  f"full={100*n_correct/n_done:.1f}% | "
                  f"func={100*n_fn/n_done:.1f}% | "
                  f"bind_tokens={result['n_binding_tokens']}/{result['n_img_tokens']}")

    # Summary
    print_summary(results, args)
    save_summary(results, args)


def print_summary(results, args):
    n = len(results)
    if n == 0:
        print("No results.")
        return

    n_correct = sum(1 for r in results if r["correct"])
    n_fn = sum(1 for r in results if r["function_match"])
    n_args = sum(1 for r in results if r["args_match"])
    n_bbox = sum(1 for r in results if r.get("bbox_match") is True)
    n_bbox_total = sum(1 for r in results if r.get("bbox_match") is not None)

    print(f"\n{'='*60}")
    print(f"Results: mode={args.mode}, alpha={args.alpha}, beta={args.beta}")
    print(f"{'='*60}")
    print(f"Total samples: {n}")
    print(f"Function accuracy: {n_fn}/{n} ({100*n_fn/n:.2f}%)")
    print(f"Full accuracy: {n_correct}/{n} ({100*n_correct/n:.2f}%)")
    if n_bbox_total > 0:
        print(f"BBox accuracy: {n_bbox}/{n_bbox_total} ({100*n_bbox/n_bbox_total:.2f}%)")

    # Per-function
    print("\nPer-function:")
    from collections import Counter
    for func, count in Counter(r["gt_function"] for r in results).most_common():
        func_results = [r for r in results if r["gt_function"] == func]
        func_correct = sum(1 for r in func_results if r["correct"])
        print(f"  {func}: {func_correct}/{count} ({100*func_correct/count:.1f}%)")


def save_summary(results, args):
    n = len(results)
    if n == 0:
        return

    summary = {
        "mode": args.mode,
        "alpha": args.alpha,
        "beta": args.beta,
        "bind_layer": args.bind_layer,
        "top_p": args.top_p,
        "layer_range": f"L{args.layer_start}-L{args.layer_end}",
        "n_samples": n,
        "full_accuracy": sum(1 for r in results if r["correct"]) / n,
        "function_accuracy": sum(1 for r in results if r["function_match"]) / n,
        "bbox_accuracy": (
            sum(1 for r in results if r.get("bbox_match") is True) /
            max(1, sum(1 for r in results if r.get("bbox_match") is not None))
        ),
        "avg_binding_tokens": np.mean([r["n_binding_tokens"] for r in results]),
        "avg_img_tokens": np.mean([r["n_img_tokens"] for r in results]),
    }

    path = os.path.join(args.output_dir, f"{args.mode}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {path}")


if __name__ == "__main__":
    main()
