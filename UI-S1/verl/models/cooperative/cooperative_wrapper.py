"""
Cooperative VLM Wrapper for Token-Level LoRA Routing.

Wraps a Qwen2.5-VL model:
  1. Replaces target nn.Linear modules with CooperativeLoRALinear
  2. Routes tokens through adapters based on type:
     - 2-agent: LoRA_V (image), LoRA_A (text/action)
     - 3-agent: LoRA_V (image), LoRA_T (thought), LoRA_A (instruction/action)
  3. Computes L_act (CE) + λ·L_bind (contrastive binding) jointly

Single forward pass — no partial forwards, no message injection.
Attention naturally bridges the adapters: q(LoRA_A) @ k(LoRA_V).
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.models.cooperative.cooperative_lora import CooperativeLoRALinear


# ── Vision token constants ──────────────────────────────────────────
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655

PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
TOKEN_PIXEL_SIZE = SPATIAL_MERGE_SIZE * PATCH_SIZE  # 28

TARGET_BBOX_RADIUS = 56  # pixels around GT coordinate

# ── Thought token bigram patterns ────────────────────────────────────
# <thought> tokenizes as two tokens in Qwen2.5-VL tokenizer:
#   13708 ("<th") + 2450 ("ought>")
# </thought> tokenizes as:
#   522 ("</") + 60565 ("thought>")
THOUGHT_OPEN_BIGRAM = (13708, 2450)
THOUGHT_CLOSE_BIGRAM = (522, 60565)


class CooperativeVLMWrapper(nn.Module):
    """Wrap a Qwen2.5-VL model with token-level cooperative LoRA.

    After wrapping, every target projection (q/k/v/o) in every transformer
    layer is a CooperativeLoRALinear. Token routing:
      - num_agents=2: image tokens -> LoRA_V, all others -> LoRA_A
      - num_agents=3: image -> LoRA_V, thought -> LoRA_T, others -> LoRA_A

    The wrapper handles:
      - Module replacement at init
      - Token mask creation and propagation
      - L_bind (contrastive binding loss) computation
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bind_weight: float = 0.1,
        bind_layer: int = 27,
        bind_temperature: float = 0.1,
        num_agents: int = 2,
        soft_routing: bool = False,
        init_sep: float = 0.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.bind_weight = bind_weight
        self.bind_layer = bind_layer
        self.bind_temperature = bind_temperature
        self.num_agents = num_agents
        self.soft_routing = soft_routing
        self.init_sep = init_sep

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.target_modules = target_modules

        # Expose base model's config and generation-related attrs for HF Trainer
        self.config = getattr(base_model, "config", None)

        # Freeze ALL base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Replace target modules with CooperativeLoRALinear
        self.coop_modules: List[CooperativeLoRALinear] = []
        self._replace_target_modules(lora_r, lora_alpha, lora_dropout,
                                     soft_routing, init_sep)

        # Generation thought state tracking (for 3-agent autoregressive decode)
        # Per-batch-element: tensors of shape [B], allocated on first use
        self._in_thought: Optional[torch.Tensor] = None   # bool [B]
        self._last_token: Optional[torch.Tensor] = None   # int64 [B]

    # ── Module replacement ──────────────────────────────────────────

    def _replace_target_modules(self, r: int, alpha: int, dropout: float,
                                soft_routing: bool = False, init_sep: float = 0.0):
        """Replace nn.Linear in each transformer layer with CooperativeLoRALinear."""
        # Find the layers ModuleList. The path depends on transformers version:
        #   Older: model.model.layers  (Qwen2_5_VLModel has layers directly)
        #   Newer (>=4.57): model.model.language_model.layers
        #       Qwen2_5_VLForConditionalGeneration.model -> Qwen2_5_VLModel
        #       Qwen2_5_VLModel.language_model -> Qwen2_5_VLTextModel
        #       Qwen2_5_VLTextModel.layers -> nn.ModuleList
        vlm = self.base_model.model  # Qwen2_5_VLModel
        if hasattr(vlm, "language_model"):
            layers = vlm.language_model.layers
        elif hasattr(vlm, "layers"):
            layers = vlm.layers
        else:
            raise AttributeError(
                f"Cannot find transformer layers in {type(vlm).__name__}. "
                f"Children: {[n for n, _ in vlm.named_children()]}"
            )

        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            for module_name in self.target_modules:
                if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    parent = layer.self_attn
                elif module_name in ("gate_proj", "up_proj", "down_proj"):
                    parent = layer.mlp
                else:
                    raise ValueError(f"Unknown target module: {module_name}")

                original = getattr(parent, module_name)
                coop_linear = CooperativeLoRALinear(
                    original, r, alpha, dropout, num_agents=self.num_agents,
                    soft_routing=soft_routing, init_sep=init_sep)
                setattr(parent, module_name, coop_linear)
                self.coop_modules.append(coop_linear)

    # ── Gradient checkpointing (delegate to base model) ────────────

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    # ── Generation with proper token routing ─────────────────────

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        """Generate with token-level LoRA routing.

        Registers a forward_pre_hook so that each internal forward call
        (prefill and every decode step) gets the correct token mask.

        For 3-agent mode, maintains a thought state machine to track whether
        we're inside a <thought>...</thought> block during autoregressive decode.
        """
        # Reset thought state for this generation call
        self._in_thought = None
        self._last_token = None

        def _pre_hook(module, args, kwargs):
            ids = kwargs.get("input_ids")
            if ids is None and len(args) > 0:
                ids = args[0]
            if ids is None:
                return

            if self.num_agents >= 3:
                if ids.shape[1] == 1:
                    # Decode step: single token per batch element
                    B = ids.shape[0]
                    mask = torch.zeros(B, 1, dtype=torch.int8, device=ids.device)
                    mask[ids[:, 0] == IMAGE_PAD_ID] = 1
                    # Per-element: set thought mask where in_thought=True and not image
                    if self._in_thought is not None:
                        thought_mask = self._in_thought & (mask[:, 0] == 0)
                        mask[thought_mask, 0] = 2
                    # Update per-element thought state
                    self._update_thought_state(ids)
                    self._set_token_mask(mask)
                else:
                    # Prefill: full sequence — scan for thought spans
                    mask = self._build_3way_mask(ids)
                    self._set_token_mask(mask)
                    # Initialize per-element thought state from prefill
                    self._init_thought_state_from_prefill(ids)
            else:
                mask = (ids == IMAGE_PAD_ID)
                self._set_token_mask(mask)

        handle = self.base_model.register_forward_pre_hook(
            _pre_hook, with_kwargs=True)
        try:
            return self.base_model.generate(input_ids=input_ids, **kwargs)
        finally:
            handle.remove()
            self._set_token_mask(None)
            self._in_thought = None
            self._last_token = None

    # ── Token mask ──────────────────────────────────────────────────

    def _set_token_mask(self, mask: Optional[torch.Tensor]):
        """Propagate token mask to all CooperativeLoRALinear modules."""
        for module in self.coop_modules:
            module.set_token_mask(mask)

    # ── 3-way mask construction ────────────────────────────────────

    def _build_3way_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build int8 mask: 0=LoRA_A, 1=LoRA_V, 2=LoRA_T.

        Detects <thought>...</thought> spans using bigram token patterns.
        <thought> tokenizes as (13708, 2450), </thought> as (522, 60565).
        """
        mask = torch.zeros_like(input_ids, dtype=torch.int8)
        mask[input_ids == IMAGE_PAD_ID] = 1  # LoRA_V

        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            seq_len = ids.shape[0]
            in_thought = False
            i = 0
            while i < seq_len - 1:
                t0 = ids[i].item()
                t1 = ids[i + 1].item()
                if t0 == THOUGHT_OPEN_BIGRAM[0] and t1 == THOUGHT_OPEN_BIGRAM[1]:
                    in_thought = True
                    mask[b, i] = 2
                    mask[b, i + 1] = 2
                    i += 2
                elif t0 == THOUGHT_CLOSE_BIGRAM[0] and t1 == THOUGHT_CLOSE_BIGRAM[1]:
                    mask[b, i] = 2
                    mask[b, i + 1] = 2
                    in_thought = False
                    i += 2
                else:
                    if in_thought and mask[b, i] != 1:
                        # Mark as thought unless it's an image token
                        mask[b, i] = 2
                    i += 1
            # Handle last token if still in thought
            if in_thought and i == seq_len - 1 and mask[b, i] != 1:
                mask[b, i] = 2

        return mask

    def _update_thought_state(self, ids: torch.Tensor):
        """Update per-batch-element thought state for single-token decode steps.

        Checks if (last_token[b], current_token[b]) forms a thought open/close
        bigram independently for each batch element.

        Args:
            ids: [B, 1] current decode tokens
        """
        B = ids.shape[0]
        current = ids[:, 0]  # [B]

        if self._last_token is None:
            self._last_token = current.clone()
            if self._in_thought is None:
                self._in_thought = torch.zeros(B, dtype=torch.bool, device=ids.device)
            return

        # Check open bigram: last == 13708 and current == 2450
        opens = ((self._last_token == THOUGHT_OPEN_BIGRAM[0])
                 & (current == THOUGHT_OPEN_BIGRAM[1]))
        # Check close bigram: last == 522 and current == 60565
        closes = ((self._last_token == THOUGHT_CLOSE_BIGRAM[0])
                  & (current == THOUGHT_CLOSE_BIGRAM[1]))

        self._in_thought[opens] = True
        self._in_thought[closes] = False
        self._last_token = current.clone()

    def _init_thought_state_from_prefill(self, ids: torch.Tensor):
        """Initialize per-batch-element thought state from the prefill sequence.

        Scans each batch element independently to determine if it ends inside
        a thought block, and records the last token ID for bigram continuation.

        Args:
            ids: [B, seq_len] prefill token IDs
        """
        B, seq_len = ids.shape
        self._in_thought = torch.zeros(B, dtype=torch.bool, device=ids.device)
        self._last_token = ids[:, -1].clone() if seq_len > 0 else torch.zeros(
            B, dtype=torch.long, device=ids.device)

        for b in range(B):
            seq = ids[b]
            in_thought = False
            for i in range(seq_len - 1):
                t0 = seq[i].item()
                t1 = seq[i + 1].item()
                if t0 == THOUGHT_OPEN_BIGRAM[0] and t1 == THOUGHT_OPEN_BIGRAM[1]:
                    in_thought = True
                elif t0 == THOUGHT_CLOSE_BIGRAM[0] and t1 == THOUGHT_CLOSE_BIGRAM[1]:
                    in_thought = False
            self._in_thought[b] = in_thought

    # ── Forward ─────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        gt_coords: Optional[list] = None,
        orig_sizes: Optional[list] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single forward pass with token-level LoRA routing.

        Args:
            input_ids:      [B, seq_len]
            attention_mask:  [B, seq_len]
            labels:          [B, seq_len] (-100 = ignore)
            gt_coords:       list of [x, y] or None per sample (for L_bind)
            orig_sizes:      list of (w, h) or None per sample (for L_bind)
            **kwargs:        pixel_values, image_grid_thw, etc.

        Returns:
            (loss, diagnostics_dict)
        """
        # Step 1: Create token mask
        if self.num_agents >= 3:
            token_mask = self._build_3way_mask(input_ids)  # [B, seq_len] int8
        else:
            token_mask = (input_ids == IMAGE_PAD_ID)  # [B, seq_len] bool

        # Step 2: Set mask on all CooperativeLoRALinear modules
        self._set_token_mask(token_mask)

        # Step 3: Determine if we need hidden states for L_bind
        need_hidden = (
            self.bind_weight > 0
            and gt_coords is not None
            and any(c is not None for c in gt_coords)
        )

        # Step 4: Standard forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=need_hidden,
            return_dict=True,
            **kwargs,
        )

        L_act = outputs.loss

        # Step 5: Binding loss
        L_bind = torch.tensor(0.0, device=L_act.device)
        bind_samples = 0
        target_sim_sum = 0.0
        nontarget_sim_sum = 0.0

        if need_hidden and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[self.bind_layer + 1]
            image_grid_thw = kwargs.get("image_grid_thw")

            L_bind_list = []
            B = input_ids.shape[0]
            for i in range(B):
                if gt_coords[i] is None:
                    continue
                if orig_sizes is not None and orig_sizes[i] is None:
                    continue
                if image_grid_thw is None:
                    continue
                try:
                    result = self._compute_bind_loss_single(
                        hidden_states[i],
                        input_ids[i],
                        image_grid_thw[i].tolist(),
                        gt_coords[i],
                        orig_sizes[i] if orig_sizes is not None else None,
                    )
                    if result is not None:
                        lb, t_sim, nt_sim = result
                        L_bind_list.append(lb)
                        target_sim_sum += t_sim
                        nontarget_sim_sum += nt_sim
                        bind_samples += 1
                except Exception:
                    pass

            if L_bind_list:
                L_bind = torch.stack(L_bind_list).mean()

        loss = L_act + self.bind_weight * L_bind

        # NOTE: Do NOT clear the token mask here. With gradient checkpointing
        # (use_reentrant=False), the recomputation pass during backward() will
        # call CooperativeLoRALinear.forward() again and needs the mask to still
        # be set. The mask is a small bool tensor — safe to keep around.
        # It will be overwritten on the next forward() call.

        diagnostics = {
            "L_act": L_act.detach(),
            "L_bind": L_bind.detach() if isinstance(L_bind, torch.Tensor) else torch.tensor(L_bind),
            "loss": loss.detach(),
            "bind_samples": bind_samples,
        }
        if bind_samples > 0:
            diagnostics["target_sim"] = target_sim_sum / bind_samples
            diagnostics["nontarget_sim"] = nontarget_sim_sum / bind_samples

        return loss, diagnostics

    # ── Binding loss ────────────────────────────────────────────────

    def _compute_bind_loss_single(
        self,
        hs: torch.Tensor,
        input_ids: torch.Tensor,
        grid_thw: list,
        gt_coord: list,
        orig_size: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[torch.Tensor, float, float]]:
        """Contrastive binding loss for one sample.

        Adapted from bind_auxiliary_train.py._compute_bind_loss_single().

        Args:
            hs:        [seq_len, D] hidden states at bind_layer
            input_ids: [seq_len] token IDs
            grid_thw:  [t, h, w] image grid dimensions (pre-merge)
            gt_coord:  [x, y] ground-truth click coordinate (original pixels)
            orig_size: (width, height) of original image

        Returns:
            (L_bind, target_sim, nontarget_sim) or None
        """
        t, h, w = grid_thw
        token_h = h // SPATIAL_MERGE_SIZE
        token_w = w // SPATIAL_MERGE_SIZE
        n_image_tokens = token_h * token_w

        resized_h = h * PATCH_SIZE
        resized_w = w * PATCH_SIZE

        # Find image token span in input_ids
        ids_list = input_ids.tolist()
        img_start = img_end = None
        for j, tok in enumerate(ids_list):
            if tok == VISION_START_ID and img_start is None:
                img_start = j
            if tok == VISION_END_ID:
                img_end = j
        if img_start is None or img_end is None:
            return None

        img_token_start = img_start + 1
        img_token_end = img_end  # exclusive

        actual_n = img_token_end - img_token_start
        if actual_n != n_image_tokens:
            return None

        # Map GT coordinate -> target token indices
        # Build per-token bboxes in resized image space
        gt_bbox = {
            "left": gt_coord[0] - TARGET_BBOX_RADIUS,
            "top": gt_coord[1] - TARGET_BBOX_RADIUS,
            "right": gt_coord[0] + TARGET_BBOX_RADIUS,
            "bottom": gt_coord[1] + TARGET_BBOX_RADIUS,
        }

        # Scale GT bbox from original to resized image space
        if orig_size is not None:
            orig_w, orig_h = orig_size
            scale_w = resized_w / orig_w
            scale_h = resized_h / orig_h
        else:
            scale_w = scale_h = 1.0

        bl = gt_bbox["left"] * scale_w
        bt = gt_bbox["top"] * scale_h
        br = gt_bbox["right"] * scale_w
        bb = gt_bbox["bottom"] * scale_h

        # Find overlapping tokens
        target_local = []
        for idx in range(n_image_tokens):
            row = idx // token_w
            col = idx % token_w
            x1 = col * TOKEN_PIXEL_SIZE
            y1 = row * TOKEN_PIXEL_SIZE
            x2 = x1 + TOKEN_PIXEL_SIZE
            y2 = y1 + TOKEN_PIXEL_SIZE
            if x2 > bl and x1 < br and y2 > bt and y1 < bb:
                target_local.append(idx)

        if not target_local:
            return None

        # Map to sequence positions
        target_seq = [img_token_start + idx for idx in target_local]
        target_set = set(target_seq)
        all_img_seq = list(range(img_token_start, img_token_end))
        nontarget_seq = [p for p in all_img_seq if p not in target_set]

        if not nontarget_seq:
            return None

        # Find task text tokens (between instruction marker and history marker)
        task_indices = self._find_task_text_indices(ids_list)
        if not task_indices:
            return None

        # Compute contrastive loss
        target_mean = hs[target_seq].mean(dim=0)
        nontarget_mean = hs[nontarget_seq].mean(dim=0)
        task_mean = hs[task_indices].mean(dim=0)

        target_sim = F.cosine_similarity(
            target_mean.unsqueeze(0), task_mean.unsqueeze(0)
        )
        nontarget_sim = F.cosine_similarity(
            nontarget_mean.unsqueeze(0), task_mean.unsqueeze(0)
        )

        logit_target = target_sim / self.bind_temperature
        logit_nontarget = nontarget_sim / self.bind_temperature
        L_bind = -torch.log(
            torch.exp(logit_target)
            / (torch.exp(logit_target) + torch.exp(logit_nontarget))
        )

        return L_bind, target_sim.detach().item(), nontarget_sim.detach().item()

    @staticmethod
    def _find_task_text_indices(ids_list: list) -> list:
        """Find task instruction token indices.

        Looks for tokens between the last VISION_END_ID and a heuristic
        end marker. Falls back to tokens between last VISION_END and
        the end of user turn.
        """
        # Find last vision_end position
        last_vision_end = None
        for j in range(len(ids_list) - 1, -1, -1):
            if ids_list[j] == VISION_END_ID:
                last_vision_end = j
                break
        if last_vision_end is None:
            return []

        # Find end of task text: next special token after vision_end
        # In Qwen2.5-VL, the assistant turn starts with <|im_start|>assistant
        # <|im_start|> = 151644, <|im_end|> = 151645
        IM_END_ID = 151645
        task_end = len(ids_list)
        for j in range(last_vision_end + 1, len(ids_list)):
            if ids_list[j] == IM_END_ID:
                task_end = j
                break

        task_start = last_vision_end + 1
        if task_start >= task_end:
            return []

        return list(range(task_start, task_end))

    # ── Trainable parameters ────────────────────────────────────────

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return all trainable parameters (LoRA_V + LoRA_A + LoRA_T if 3-agent)."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Save / load ─────────────────────────────────────────────────

    def save_cooperative_checkpoint(self, output_dir: str):
        """Save adapter weights separately: lora_v.pt, lora_a.pt, (lora_t.pt)."""
        os.makedirs(output_dir, exist_ok=True)
        v_state, a_state, t_state = {}, {}, {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A_v" in name or "lora_B_v" in name:
                v_state[name] = param.data.clone().cpu()
            elif "lora_A_a" in name or "lora_B_a" in name:
                a_state[name] = param.data.clone().cpu()
            elif "lora_A_t" in name or "lora_B_t" in name:
                t_state[name] = param.data.clone().cpu()

        torch.save(v_state, os.path.join(output_dir, "lora_v.pt"))
        torch.save(a_state, os.path.join(output_dir, "lora_a.pt"))
        if t_state:
            torch.save(t_state, os.path.join(output_dir, "lora_t.pt"))

        # Save sep params for soft routing
        sep_state = {}
        if self.soft_routing:
            for name, param in self.named_parameters():
                if name.endswith(".sep") and param.requires_grad:
                    sep_state[name] = param.data.clone().cpu()
            if sep_state:
                torch.save(sep_state, os.path.join(output_dir, "lora_sep.pt"))

        # Save config
        config = {
            "target_modules": self.target_modules,
            "bind_weight": self.bind_weight,
            "bind_layer": self.bind_layer,
            "bind_temperature": self.bind_temperature,
            "num_agents": self.num_agents,
            "lora_v_params": sum(v.numel() for v in v_state.values()),
            "lora_a_params": sum(v.numel() for v in a_state.values()),
            "soft_routing": self.soft_routing,
            "init_sep": self.init_sep,
        }
        if t_state:
            config["lora_t_params"] = sum(v.numel() for v in t_state.values())
        if sep_state:
            config["sep_values"] = {
                name: torch.sigmoid(param).item()
                for name, param in sep_state.items()
            }
        with open(os.path.join(output_dir, "cooperative_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_cooperative_checkpoint(self, checkpoint_dir: str):
        """Load adapter weights: lora_v.pt, lora_a.pt, (lora_t.pt)."""
        v_path = os.path.join(checkpoint_dir, "lora_v.pt")
        a_path = os.path.join(checkpoint_dir, "lora_a.pt")
        t_path = os.path.join(checkpoint_dir, "lora_t.pt")

        if os.path.exists(v_path):
            v_state = torch.load(v_path, map_location="cpu", weights_only=True)
            missing = []
            for name, param in self.named_parameters():
                if name in v_state:
                    param.data.copy_(v_state[name].to(param.device))
                elif "lora_A_v" in name or "lora_B_v" in name:
                    missing.append(name)
            if missing:
                print(f"Warning: {len(missing)} LoRA_V params not found in checkpoint")

        if os.path.exists(a_path):
            a_state = torch.load(a_path, map_location="cpu", weights_only=True)
            missing = []
            for name, param in self.named_parameters():
                if name in a_state:
                    param.data.copy_(a_state[name].to(param.device))
                elif "lora_A_a" in name or "lora_B_a" in name:
                    missing.append(name)
            if missing:
                print(f"Warning: {len(missing)} LoRA_A params not found in checkpoint")

        if os.path.exists(t_path) and self.num_agents >= 3:
            t_state = torch.load(t_path, map_location="cpu", weights_only=True)
            missing = []
            for name, param in self.named_parameters():
                if name in t_state:
                    param.data.copy_(t_state[name].to(param.device))
                elif "lora_A_t" in name or "lora_B_t" in name:
                    missing.append(name)
            if missing:
                print(f"Warning: {len(missing)} LoRA_T params not found in checkpoint")

        # Load sep params for soft routing
        sep_path = os.path.join(checkpoint_dir, "lora_sep.pt")
        if os.path.exists(sep_path) and self.soft_routing:
            sep_state = torch.load(sep_path, map_location="cpu", weights_only=True)
            loaded = 0
            for name, param in self.named_parameters():
                if name in sep_state:
                    param.data.copy_(sep_state[name].to(param.device))
                    loaded += 1
            if loaded > 0:
                print(f"Loaded {loaded} sep params from checkpoint")
