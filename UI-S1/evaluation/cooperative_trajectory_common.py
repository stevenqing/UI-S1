"""Shared utilities for HF-based cooperative LoRA trajectory evaluation.

Used by:
  - eval_cooperative_ac_trajectory.py     (AndroidControl)
  - eval_cooperative_odyssey_trajectory.py (GUI-Odyssey)

Provides:
  - load_cooperative_model: build CooperativeVLMWrapper from base + ckpt
  - jsonformat_messages_to_qwen: convert JsonFormat-style messages to Qwen
    processor format and load images.
  - cooperative_generate: run a single HF generate() call against the wrapper.
  - safe_parse_response: JSON-tolerant <think>/<action> parser.
  - length_bucket / compute_trajectory_metrics / _json_default: misc helpers.
"""

import copy
import json
import os
import re
import sys

import torch
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from verl.models.cooperative.cooperative_wrapper import CooperativeVLMWrapper


# ── Model loading ─────────────────────────────────────────────────────

def load_cooperative_model(base_model_path, coop_checkpoint_path, device):
    """Load Qwen2.5-VL base + cooperative LoRA wrapper from a checkpoint.

    Mirrors evaluation/eval_cooperative_batch.py:load_model() so the v6.5
    ckpts continue to work as-is.
    """
    print(f"[{device}] Loading base model from {base_model_path}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )

    config_path = os.path.join(coop_checkpoint_path, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)

    lora_v = torch.load(
        os.path.join(coop_checkpoint_path, "lora_v.pt"),
        map_location="cpu",
        weights_only=True,
    )
    r = [v for k, v in lora_v.items() if "lora_A_v" in k][0].shape[0]

    num_agents = coop_config.get("num_agents", 2)
    if num_agents == 2 and os.path.exists(os.path.join(coop_checkpoint_path, "lora_t.pt")):
        num_agents = 3
        print(f"[{device}] Detected lora_t.pt — using num_agents=3")

    soft_routing = coop_config.get("soft_routing", False)
    init_sep = coop_config.get("init_sep", 0.0)
    cooperative_comm = coop_config.get("cooperative_comm", False)
    gate_init = coop_config.get("gate_init", -3.0)
    gate_type = coop_config.get("gate_type", "sigmoid")
    routing_mode = coop_config.get("routing_mode", "hard")

    print(
        f"[{device}] Wrapping cooperative LoRA "
        f"(r={r}, targets={coop_config['target_modules']}, "
        f"num_agents={num_agents}, gate_type={gate_type}, "
        f"cooperative_comm={cooperative_comm}, routing_mode={routing_mode})"
    )
    model = CooperativeVLMWrapper(
        base_model=base_model,
        lora_r=r,
        lora_alpha=r * 2,
        lora_dropout=0.0,
        target_modules=coop_config["target_modules"],
        bind_weight=0.0,
        num_agents=num_agents,
        soft_routing=soft_routing,
        init_sep=init_sep,
        cooperative_comm=cooperative_comm,
        gate_init=gate_init,
        gate_type=gate_type,
        routing_mode=routing_mode,
    )
    model.load_cooperative_checkpoint(coop_checkpoint_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    return model, processor, base_model.device


# ── Message construction ──────────────────────────────────────────────

def jsonformat_messages_to_qwen(messages):
    """Convert JsonFormat-style messages to Qwen processor input.

    JsonFormat content blocks look like:
        {"text": "..."}
        {"image": "/abs/path", "width":..., "height":..., "resized_width":..., "resized_height":...}

    Qwen processor expects:
        {"type": "text", "text": "..."}
        {"type": "image", "image": <PIL.Image>}

    Returns:
        (qwen_messages, pil_images)
        qwen_messages : list of {role, content}
        pil_images    : list of PIL.Image, in the same order as image content blocks
    """
    qwen_messages = []
    pil_images = []
    for msg in messages:
        new_content = []
        for block in msg["content"]:
            if isinstance(block, str):
                new_content.append({"type": "text", "text": block})
                continue
            if "text" in block:
                new_content.append({"type": "text", "text": block["text"]})
            elif "image" in block:
                img = block["image"]
                if isinstance(img, Image.Image):
                    pil = img
                else:
                    pil = Image.open(img).convert("RGB")
                pil_images.append(pil)
                new_content.append({"type": "image", "image": pil})
        qwen_messages.append({"role": msg["role"], "content": new_content})
    return qwen_messages, pil_images


@torch.no_grad()
def cooperative_generate(
    model,
    processor,
    messages,
    device,
    max_new_tokens=512,
    do_sample=False,
):
    """Run a single HF generate() call through the cooperative wrapper.

    Args:
        messages: JsonFormat-style message list (with {"text"} / {"image"} blocks)
                  or already in Qwen processor format. The function auto-detects.
    Returns:
        Decoded response string (no special tokens).
    """
    needs_conversion = any(
        any(("type" not in b) for b in msg["content"] if isinstance(b, dict))
        for msg in messages
    )
    if needs_conversion:
        qwen_messages, pil_images = jsonformat_messages_to_qwen(messages)
    else:
        # Already in Qwen format — extract PIL images
        qwen_messages = messages
        pil_images = []
        for msg in qwen_messages:
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "image":
                    img = block["image"]
                    if isinstance(img, str):
                        img = Image.open(img).convert("RGB")
                    pil_images.append(img)

    text = processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True,
    )

    if pil_images:
        inputs = processor(
            text=[text], images=pil_images,
            return_tensors="pt", padding=True,
        )
    else:
        inputs = processor(
            text=[text],
            return_tensors="pt", padding=True,
        )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    response = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True,
    )
    return response


# ── Response parsing ──────────────────────────────────────────────────

def safe_parse_response(fm, model_response):
    """Parse model response with JSON error tolerance.

    Mirrors gui_odyssey_eval/eval_ar_trajectory.py:safe_parse_response().
    """
    try:
        return fm.parse_response(model_response)
    except json.JSONDecodeError:
        fixed = re.sub(r"\}\}(\s*</action>)", r"}\1", model_response)
        if fixed != model_response:
            try:
                return fm.parse_response(fixed)
            except Exception:
                pass
        match = re.search(r"<action>\s*(\{.*?\})\s*</action>", model_response, re.DOTALL)
        if match:
            action_str = match.group(1)
            while action_str.endswith("}}"):
                action_str = action_str[:-1]
            try:
                action_content = json.loads(action_str)
                think_match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
                return {
                    "think": think_match.group(1).strip() if think_match else "",
                    "action": action_str,
                    "action_content": action_content,
                }
            except json.JSONDecodeError:
                pass
        raise


# ── Misc helpers ──────────────────────────────────────────────────────

def length_bucket(n):
    if n <= 3:
        return "short(1-3)"
    elif n <= 7:
        return "medium(4-7)"
    elif n <= 15:
        return "long(8-15)"
    else:
        return "vlong(16+)"


def compute_trajectory_metrics(results):
    """Standard trajectory metrics: TSR, avg_progress, scattered_progress."""
    if not results:
        return {
            "tsr": 0.0,
            "avg_progress": 0.0,
            "scattered_progress": 0.0,
            "n": 0,
            "success_count": 0,
        }
    n = len(results)
    success_count = sum(1 for r in results if r["task_success"])
    tsr = success_count / n
    progresses = [r["final_step_id"] / r["num_steps"] for r in results if r["num_steps"] > 0]
    avg_progress = sum(progresses) / len(progresses) if progresses else 0.0
    total_steps = sum(r["num_steps"] for r in results)
    total_correct = sum(r["final_step_id"] for r in results)
    scattered_progress = total_correct / total_steps if total_steps > 0 else 0.0
    return {
        "tsr": tsr,
        "avg_progress": avg_progress,
        "scattered_progress": scattered_progress,
        "n": n,
        "success_count": success_count,
    }


def _json_default(obj):
    """JSON serializer fallback for numpy types."""
    try:
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def shard_episodes(episodes, shard_id, num_shards):
    """Slice episodes for a single shard. Round-robin (modulo) keeps work balanced."""
    if num_shards <= 1:
        return episodes
    return [ep for i, ep in enumerate(episodes) if i % num_shards == shard_id]


def aggregate_action_stats(results):
    """Per-action-type breakdown of type/extract match rates."""
    action_stats = {}
    for r in results:
        for s in r["step_results"]:
            at = s["gt_action_type"]
            if at not in action_stats:
                action_stats[at] = {"total": 0, "type_match": 0, "extract_match": 0}
            action_stats[at]["total"] += 1
            action_stats[at]["type_match"] += int(s["type_match"])
            action_stats[at]["extract_match"] += int(s["extract_match"])
    for at in action_stats:
        t = action_stats[at]["total"]
        action_stats[at]["type_match_rate"] = action_stats[at]["type_match"] / t if t > 0 else 0
        action_stats[at]["extract_match_rate"] = action_stats[at]["extract_match"] / t if t > 0 else 0
    return action_stats
