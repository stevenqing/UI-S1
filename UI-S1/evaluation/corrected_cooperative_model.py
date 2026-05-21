"""Corrected Cooperative Model: lossless inference from merged model + correction adapters.

After merging LoRA_A into base weights, image tokens need a correction:
  y_image = W_merged @ x + scaling * C_out @ (C_in @ x)

This module wraps the merged Qwen2.5-VL with CorrectedLinear layers that
apply corrections at image token positions during prefill.

During decode (all text tokens), no correction is applied → same speed as
a standard merged model.

Usage:
    from evaluation.corrected_cooperative_model import (
        load_corrected_model, corrected_generate)

    model, processor = load_corrected_model("/tmp/merged_lossless", device="cuda:0")
    response = corrected_generate(model, processor, messages, device="cuda:0")
"""

import json
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

IMAGE_PAD_ID = 151655


# ── CorrectedLinear ───────────────────────────────────────────────────

class CorrectedLinear(nn.Module):
    """Linear layer with low-rank correction at image token positions.

    y = base_linear(x)                                       for text tokens
    y = base_linear(x) + scaling * C_out @ (C_in @ x)        for image tokens

    The base_linear already has LoRA_A merged into its weights.
    The correction restores the exact LoRA_V behavior for image tokens.
    """

    def __init__(self, base_linear: nn.Linear, C_in: torch.Tensor,
                 C_out: torch.Tensor, scaling: float):
        super().__init__()
        self.base_linear = base_linear
        self.register_buffer("C_in", C_in)    # [2r, d_in]
        self.register_buffer("C_out", C_out)  # [d_out, 2r]
        self.scaling = scaling
        self._image_mask: Optional[torch.Tensor] = None

    def set_image_mask(self, mask: Optional[torch.Tensor]):
        """Set [B, S] bool mask: True = image token (needs correction)."""
        self._image_mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_linear(x)
        mask = self._image_mask
        if mask is None or not mask.any():
            return y
        # Broadcast correction: compute for all positions, zero at text positions
        mask_f = mask.unsqueeze(-1).to(y.dtype)       # [B, S, 1]
        h = F.linear(x, self.C_in)                     # [B, S, 2r]
        corr = F.linear(h, self.C_out) * self.scaling   # [B, S, d_out]
        return y + corr * mask_f


# ── CorrectedCooperativeModel ────────────────────────────────────────

class CorrectedCooperativeModel(nn.Module):
    """Qwen2.5-VL with merged LoRA_A weights + lossless correction adapters.

    Text tokens:  use merged weights (exact, zero overhead).
    Image tokens: use merged weights + low-rank correction (exact).
    Decode phase: always text → no correction, standard speed.
    """

    def __init__(self, base_model: Qwen2_5_VLForConditionalGeneration,
                 corrections: dict, scaling: float, target_modules: List[str]):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.corrected_modules: List[CorrectedLinear] = []

        # Find transformer layers
        vlm = base_model.model
        if hasattr(vlm, "language_model"):
            layers = vlm.language_model.layers
        elif hasattr(vlm, "layers"):
            layers = vlm.layers
        else:
            raise AttributeError(
                f"Cannot find layers in {type(vlm).__name__}. "
                f"Children: {[n for n, _ in vlm.named_children()]}"
            )

        # Build mapping: base_key → {C_in, C_out}
        corr_map = {}
        for key, val in corrections.items():
            if key.endswith(".C_in"):
                bk = key[: -len(".C_in")]
                corr_map.setdefault(bk, {})["C_in"] = val
            elif key.endswith(".C_out"):
                bk = key[: -len(".C_out")]
                corr_map.setdefault(bk, {})["C_out"] = val

        # Replace target modules with CorrectedLinear
        replaced = 0
        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            for module_name in target_modules:
                if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    parent = layer.self_attn
                    bk = f"model.layers.{layer_idx}.self_attn.{module_name}.weight"
                elif module_name in ("gate_proj", "up_proj", "down_proj"):
                    parent = layer.mlp
                    bk = f"model.layers.{layer_idx}.mlp.{module_name}.weight"
                else:
                    continue

                if bk not in corr_map:
                    continue

                original = getattr(parent, module_name)
                device = original.weight.device
                corrected = CorrectedLinear(
                    original,
                    corr_map[bk]["C_in"].to(device),
                    corr_map[bk]["C_out"].to(device),
                    scaling,
                )
                setattr(parent, module_name, corrected)
                self.corrected_modules.append(corrected)
                replaced += 1

        print(f"Replaced {replaced} modules with CorrectedLinear")

    def _set_image_mask(self, mask: Optional[torch.Tensor]):
        for m in self.corrected_modules:
            m.set_image_mask(mask)

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        """Generate with lossless image-token corrections.

        Registers a forward_pre_hook so that each internal forward call
        (prefill + every decode step) gets the correct image mask.
        During decode, input_ids is [B,1] of text tokens → mask is all-False
        → no correction applied → standard merged-model speed.
        """

        def _pre_hook(module, args, kwargs):
            ids = kwargs.get("input_ids")
            if ids is None and len(args) > 0:
                ids = args[0]
            if ids is None:
                return
            mask = (ids == IMAGE_PAD_ID)
            self._set_image_mask(mask)

        handle = self.base_model.register_forward_pre_hook(
            _pre_hook, with_kwargs=True
        )
        try:
            return self.base_model.generate(input_ids=input_ids, **kwargs)
        finally:
            handle.remove()
            self._set_image_mask(None)


# ── Loading ───────────────────────────────────────────────────────────

def load_corrected_model(merged_dir, device="cuda:0"):
    """Load merged model + correction adapters → CorrectedCooperativeModel.

    Args:
        merged_dir: directory produced by merge_cooperative_lossless.py
        device: device string or device_map dict

    Returns:
        (model, processor)
    """
    corr_config_path = os.path.join(merged_dir, "correction_config.json")
    with open(corr_config_path) as f:
        corr_config = json.load(f)

    print(f"Loading merged model from {merged_dir}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        merged_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )

    corr_path = os.path.join(merged_dir, "correction_adapters.pt")
    corrections = torch.load(corr_path, map_location="cpu", weights_only=True)
    print(f"Loaded {len(corrections)} correction tensors")

    model = CorrectedCooperativeModel(
        base_model=base_model,
        corrections=corrections,
        scaling=corr_config["scaling"],
        target_modules=corr_config["target_modules"],
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(merged_dir, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    return model, processor


# ── Generation helper ─────────────────────────────────────────────────

@torch.no_grad()
def corrected_generate(model, processor, messages, device,
                       max_new_tokens=512, do_sample=False):
    """Run generate() through the corrected model.

    Same signature as cooperative_trajectory_common.cooperative_generate()
    so eval scripts can swap between them.
    """
    # Auto-detect message format
    needs_conversion = any(
        any(("type" not in b) for b in msg["content"] if isinstance(b, dict))
        for msg in messages
    )
    if needs_conversion:
        from evaluation.cooperative_trajectory_common import jsonformat_messages_to_qwen
        qwen_messages, pil_images = jsonformat_messages_to_qwen(messages)
    else:
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
        inputs = processor(text=[text], return_tensors="pt", padding=True)

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
    )
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
