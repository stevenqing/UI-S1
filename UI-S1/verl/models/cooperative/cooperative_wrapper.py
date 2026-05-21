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

# ── Coordinate token constants (for coord_routing) ────────────────────
# In Qwen2.5-VL tokenizer:
#   "coordinate" → single token 62526
#   "bbox"       → single token 58456
#   digits 0-9   → ids 15-24
#   "."          → id 13
#   ","          → id 11
#   " "          → id 220
#   " ["         → id 508  (array open)
#   "],"         → id 1125 (array close mid-JSON)
#   "]}"         → id 81136 (array close end-JSON)
COORD_KEY_ID = 62526       # 'coordinate'
BBOX_KEY_ID = 58456        # 'bbox'
DIGIT_TOKEN_IDS = set(range(15, 25))  # 0-9
COORD_PUNCT_IDS = {13, 11, 220}      # '.', ',', ' '
COORD_ALL_VALUE_IDS = DIGIT_TOKEN_IDS | COORD_PUNCT_IDS
BRACKET_OPEN_ID = 508      # ' ['
BRACKET_CLOSE_IDS = {1125, 81136}  # '],', ']}'

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
        cooperative_comm: bool = False,
        gate_init: float = -3.0,
        gate_type: str = "sigmoid",
        routing_mode: str = "hard",
        coord_routing: bool = False,
        coord_only_routing: bool = False,
        coop_reasoning_alpha: float = 0.0,
        balance_weight: float = 0.0,
        shared_B: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.bind_weight = bind_weight
        self.bind_layer = bind_layer
        self.bind_temperature = bind_temperature
        self.num_agents = num_agents
        self.soft_routing = soft_routing
        self.init_sep = init_sep
        self.cooperative_comm = cooperative_comm
        self.gate_init = gate_init
        self.gate_type = gate_type
        self.coord_routing = coord_routing
        self.coord_only_routing = coord_only_routing
        self.coop_reasoning_alpha = coop_reasoning_alpha
        self.balance_weight = balance_weight
        self.shared_B = shared_B
        if routing_mode not in ("hard", "merge", "learned"):
            raise ValueError(
                f"routing_mode must be 'hard', 'merge', or 'learned', got {routing_mode}")
        self.routing_mode = routing_mode

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.target_modules = target_modules

        # Expose base model's config and generation-related attrs for HF Trainer
        self.config = getattr(base_model, "config", None)

        # Freeze ALL base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Per-layer learned routers (v8). Created only in learned mode.
        # Each router is nn.Linear(hidden_size, 1), shared across q/k/v/o
        # within one transformer layer.
        self.routers: Optional[nn.ModuleList] = None
        if self.routing_mode == "learned":
            self._create_routers()

        # Replace target modules with CooperativeLoRALinear
        self.coop_modules: List[CooperativeLoRALinear] = []
        # Track layer index for each coop module, so we can map module → router
        self._module_to_layer: List[int] = []
        self._replace_target_modules(lora_r, lora_alpha, lora_dropout,
                                     soft_routing, init_sep)

        if self.routing_mode == "learned":
            print(f"[CooperativeVLMWrapper] routing_mode=learned: "
                  f"{len(self.routers)} per-layer routers created, "
                  f"balance_weight={balance_weight}")

        # Generation thought state tracking (for 3-agent autoregressive decode)
        # Per-batch-element: tensors of shape [B], allocated on first use
        self._in_thought: Optional[torch.Tensor] = None   # bool [B]
        self._last_token: Optional[torch.Tensor] = None   # int64 [B]

        # Generation coord state tracking (for coord_routing decode)
        # Tracks whether we're inside a "coordinate": [...] or "bbox": [...] region
        self._in_coord_region: Optional[torch.Tensor] = None  # bool [B]
        self._seen_bracket: Optional[torch.Tensor] = None     # bool [B]

        if coord_only_routing:
            print(f"[CooperativeVLMWrapper] coord_only_routing=True (v10): "
                  f"ONLY coordinate/bbox digit tokens → LoRA_V, "
                  f"all other tokens (including image) → LoRA_A")
        elif coord_routing:
            print(f"[CooperativeVLMWrapper] coord_routing=True: "
                  f"coordinate/bbox digit tokens will be routed to LoRA_V")
        if coop_reasoning_alpha > 0:
            print(f"[CooperativeVLMWrapper] coop_reasoning_alpha={coop_reasoning_alpha}: "
                  f"assistant tokens get α·V + (1-α)·A cooperative reasoning")

    # ── Module replacement ──────────────────────────────────────────

    def _get_transformer_layers(self):
        """Locate the transformer layer ModuleList (handles transformers version skew)."""
        vlm = self.base_model.model  # Qwen2_5_VLModel
        if hasattr(vlm, "language_model"):
            return vlm.language_model.layers
        elif hasattr(vlm, "layers"):
            return vlm.layers
        else:
            raise AttributeError(
                f"Cannot find transformer layers in {type(vlm).__name__}. "
                f"Children: {[n for n, _ in vlm.named_children()]}"
            )

    def _create_routers(self):
        """Create per-layer nn.Linear(hidden_size, 1) routers for learned routing."""
        layers = self._get_transformer_layers()
        num_layers = len(layers)
        # Get hidden size from the first q_proj we find
        hidden_size = layers[0].self_attn.q_proj.in_features
        # Device from model
        device = next(self.base_model.parameters()).device
        dtype = next(self.base_model.parameters()).dtype

        self.routers = nn.ModuleList()
        for _ in range(num_layers):
            r = nn.Linear(hidden_size, 1, bias=True)
            # Init: weight=0, bias=0 -> sigmoid(0) = 0.5 (uniform blend).
            # Warm-start (if enabled) will overwrite these before training.
            nn.init.zeros_(r.weight)
            nn.init.zeros_(r.bias)
            r = r.to(device=device, dtype=dtype)
            # Routers are trainable
            for p in r.parameters():
                p.requires_grad = True
            self.routers.append(r)

    def _replace_target_modules(self, r: int, alpha: int, dropout: float,
                                soft_routing: bool = False, init_sep: float = 0.0):
        """Replace nn.Linear in each transformer layer with CooperativeLoRALinear.

        In learned mode, attaches the per-layer router to every module in that layer.
        """
        layers = self._get_transformer_layers()

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
                    soft_routing=soft_routing, init_sep=init_sep,
                    cooperative_comm=self.cooperative_comm,
                    gate_init=self.gate_init,
                    gate_type=self.gate_type,
                    routing_mode=self.routing_mode,
                    shared_B=self.shared_B)
                # Attach shared per-layer router in learned mode.
                # Only modules whose in_features matches hidden_size get the
                # router. Modules with a different input dim (e.g. down_proj
                # which takes intermediate_size=18944) fall back to merge
                # (50/50 blend) inside CooperativeLoRALinear.forward().
                if self.routing_mode == "learned" and self.routers is not None:
                    if original.in_features == self.routers[layer_idx].in_features:
                        coop_linear.set_router(self.routers[layer_idx])
                setattr(parent, module_name, coop_linear)
                self.coop_modules.append(coop_linear)
                self._module_to_layer.append(layer_idx)

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
        # Reset coord state
        self._in_coord_region = None
        self._seen_bracket = None

        def _pre_hook(module, args, kwargs):
            ids = kwargs.get("input_ids")
            if ids is None and len(args) > 0:
                ids = args[0]
            if ids is None:
                return

            # v7 merge mode: no routing — leave mask cleared so forward()
            # takes the merge path via routing_mode check.
            if self.routing_mode == "merge":
                self._set_token_mask(None)
                return

            # v8 learned mode: routing is computed inside each module from
            # hidden states, no mask needed.
            if self.routing_mode == "learned":
                self._set_token_mask(None)
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
            elif self.coop_reasoning_alpha > 0:
                # 2-agent with cooperative reasoning: float mask
                if ids.shape[1] == 1:
                    # Decode: all tokens are assistant → α, unless image/coord → 1.0
                    mask = torch.full(
                        ids.shape, self.coop_reasoning_alpha,
                        dtype=torch.float32, device=ids.device)
                    mask[ids == IMAGE_PAD_ID] = 1.0
                    if self.coord_routing:
                        coord_mask = self._update_coord_state(ids)  # [B, 1] bool
                        mask[coord_mask] = 1.0
                    self._set_token_mask(mask)
                else:
                    # Prefill: float mask with assistant span detection
                    mask = torch.zeros(
                        ids.shape, dtype=torch.float32, device=ids.device)
                    mask[ids == IMAGE_PAD_ID] = 1.0
                    if self.coord_routing:
                        self._mark_coord_tokens(mask, ids)
                        self._init_coord_state_from_prefill(ids)
                    self._mark_assistant_spans(mask, ids, self.coop_reasoning_alpha)
                    self._set_token_mask(mask)
            elif self.coord_only_routing:
                # v10: ONLY coordinate digits → V, everything else → A
                if ids.shape[1] == 1:
                    # Decode step
                    coord_mask = self._update_coord_state(ids)  # [B, 1] bool
                    self._set_token_mask(coord_mask)
                else:
                    # Prefill
                    mask = torch.zeros(
                        ids.shape, dtype=torch.bool, device=ids.device)
                    self._mark_coord_tokens(mask, ids)
                    self._init_coord_state_from_prefill(ids)
                    self._set_token_mask(mask)
            else:
                if ids.shape[1] == 1 and self.coord_routing:
                    # Decode step with coord routing
                    mask = (ids == IMAGE_PAD_ID)
                    coord_mask = self._update_coord_state(ids)  # [B, 1] bool
                    mask = mask | coord_mask
                    self._set_token_mask(mask)
                else:
                    # Prefill or no coord_routing
                    mask = (ids == IMAGE_PAD_ID)
                    if self.coord_routing:
                        self._mark_coord_tokens(mask, ids)
                        self._init_coord_state_from_prefill(ids)
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
            self._in_coord_region = None
            self._seen_bracket = None

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

    # ── Coord-aware mask construction (for coord_routing) ────────────

    # Token IDs that can appear between a coord key and the opening bracket:
    #   788  = '":' (colon after key)
    #   15-24 = digits (part of key name like "coordinate2")
    #   1    = '"' (closing quote of key)
    #   330  = ' "' (space+quote, if tokenizer splits differently)
    _COORD_KEY_TRAIL_IDS = {788, 1, 330} | DIGIT_TOKEN_IDS

    def _mark_coord_tokens(self, mask: torch.Tensor, input_ids: torch.Tensor):
        """In-place: set mask=True for coordinate/bbox digit tokens.

        Scans for 'coordinate' (62526) or 'bbox' (58456) key tokens, then
        marks subsequent digit/punct tokens inside [...] as LoRA_V.

        Handles 'coordinate2' (swipe) correctly: the '2' after 'coordinate'
        is part of the key name and is skipped (not marked as LoRA_V).

        Args:
            mask: [B, seq_len] bool tensor (modified in-place)
            input_ids: [B, seq_len] token IDs
        """
        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            seq_len = ids.shape[0]
            in_coord = False
            in_bracket = False
            for i in range(seq_len):
                tid = ids[i].item()
                if not in_coord:
                    if tid == COORD_KEY_ID or tid == BBOX_KEY_ID:
                        in_coord = True
                        in_bracket = False
                else:
                    if not in_bracket:
                        if tid == BRACKET_OPEN_ID:
                            in_bracket = True
                        elif tid not in self._COORD_KEY_TRAIL_IDS:
                            in_coord = False  # unexpected token, abort
                    else:
                        if tid in BRACKET_CLOSE_IDS:
                            in_coord = False
                            in_bracket = False
                        elif tid in COORD_ALL_VALUE_IDS:
                            mask[b, i] = True
                        else:
                            # unexpected token inside bracket, abort
                            in_coord = False
                            in_bracket = False

    def _update_coord_state(self, ids: torch.Tensor) -> torch.Tensor:
        """Update coord state machine for single-token decode and return mask.

        Args:
            ids: [B, 1] current decode tokens

        Returns:
            coord_mask: [B, 1] bool — True if this token is a coord value
        """
        B = ids.shape[0]
        current = ids[:, 0]  # [B]
        coord_mask = torch.zeros(B, 1, dtype=torch.bool, device=ids.device)

        if self._in_coord_region is None:
            self._in_coord_region = torch.zeros(B, dtype=torch.bool, device=ids.device)
            self._seen_bracket = torch.zeros(B, dtype=torch.bool, device=ids.device)

        trail_ids = self._COORD_KEY_TRAIL_IDS
        for b in range(B):
            tid = current[b].item()
            if not self._in_coord_region[b]:
                if tid == COORD_KEY_ID or tid == BBOX_KEY_ID:
                    self._in_coord_region[b] = True
                    self._seen_bracket[b] = False
            else:
                if not self._seen_bracket[b]:
                    if tid == BRACKET_OPEN_ID:
                        self._seen_bracket[b] = True
                    elif tid not in trail_ids:
                        self._in_coord_region[b] = False
                else:
                    if tid in BRACKET_CLOSE_IDS:
                        self._in_coord_region[b] = False
                        self._seen_bracket[b] = False
                    elif tid in COORD_ALL_VALUE_IDS:
                        coord_mask[b, 0] = True
                    else:
                        self._in_coord_region[b] = False
                        self._seen_bracket[b] = False

        return coord_mask

    def _init_coord_state_from_prefill(self, ids: torch.Tensor):
        """Initialize coord state from prefill sequence.

        Args:
            ids: [B, seq_len] prefill token IDs
        """
        B, seq_len = ids.shape
        self._in_coord_region = torch.zeros(B, dtype=torch.bool, device=ids.device)
        self._seen_bracket = torch.zeros(B, dtype=torch.bool, device=ids.device)

        trail_ids = self._COORD_KEY_TRAIL_IDS
        for b in range(B):
            in_coord = False
            seen_bracket = False
            for i in range(seq_len):
                tid = ids[b, i].item()
                if not in_coord:
                    if tid == COORD_KEY_ID or tid == BBOX_KEY_ID:
                        in_coord = True
                        seen_bracket = False
                else:
                    if not seen_bracket:
                        if tid == BRACKET_OPEN_ID:
                            seen_bracket = True
                        elif tid not in trail_ids:
                            in_coord = False
                    else:
                        if tid in BRACKET_CLOSE_IDS:
                            in_coord = False
                            seen_bracket = False
                        elif tid not in COORD_ALL_VALUE_IDS:
                            in_coord = False
                            seen_bracket = False
            self._in_coord_region[b] = in_coord
            self._seen_bracket[b] = seen_bracket

    # ── Assistant span detection (for cooperative reasoning at inference) ──

    # Qwen2.5-VL chat template tokens
    _IM_START_ID = 151644
    _ASSISTANT_ID = 77091
    _NEWLINE_ID = 198
    _IM_END_ID = 151645

    def _mark_assistant_spans(self, mask: torch.Tensor, input_ids: torch.Tensor,
                              alpha: float):
        """In-place: set mask=alpha for assistant response tokens.

        Detects <|im_start|>assistant\\n ... <|im_end|> spans. Only sets alpha
        on positions where mask < 0.5 (i.e., not already image/coord tokens).

        Used during generate() where labels are not available.

        Args:
            mask: [B, seq_len] float tensor (modified in-place)
            input_ids: [B, seq_len] token IDs
            alpha: cooperative reasoning weight
        """
        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            seq_len = ids.shape[0]
            in_assistant = False
            i = 0
            while i < seq_len:
                tid = ids[i].item()
                if not in_assistant:
                    if (tid == self._IM_START_ID and i + 2 < seq_len and
                            ids[i + 1].item() == self._ASSISTANT_ID and
                            ids[i + 2].item() == self._NEWLINE_ID):
                        in_assistant = True
                        i += 3  # skip <|im_start|>assistant\n header
                        continue
                else:
                    if tid == self._IM_END_ID:
                        in_assistant = False
                    elif mask[b, i] < 0.5:  # not already image/coord
                        mask[b, i] = alpha
                i += 1

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
        # Step 1: Create token mask (skipped in merge/learned modes)
        if self.routing_mode == "merge":
            # No routing: every token gets 0.5 * (delta_v + delta_a).
            # Clear any stale mask so a dev-time bug can't accidentally route.
            self._set_token_mask(None)
        elif self.routing_mode == "learned":
            # v8: router reads from hidden states directly, no token_mask needed.
            self._set_token_mask(None)
        else:
            if self.num_agents >= 3:
                token_mask = self._build_3way_mask(input_ids)  # [B, seq_len] int8
            elif self.coop_reasoning_alpha > 0:
                # Cooperative reasoning: float mask with α-mixing on assistant tokens
                token_mask = torch.zeros(
                    input_ids.shape, dtype=torch.float32, device=input_ids.device)
                token_mask[input_ids == IMAGE_PAD_ID] = 1.0
                if self.coord_routing:
                    self._mark_coord_tokens(token_mask, input_ids)
                # Mark assistant tokens (labels != -100) with α
                if labels is not None:
                    is_assistant = (labels != -100)
                    is_v = (token_mask > 0.5)
                    is_reasoning = is_assistant & ~is_v
                    token_mask[is_reasoning] = self.coop_reasoning_alpha
            elif self.coord_only_routing:
                # v10: ONLY coordinate digits → LoRA_V, everything else → LoRA_A
                # (no image token routing — image tokens go through LoRA_A)
                token_mask = torch.zeros(
                    input_ids.shape, dtype=torch.bool, device=input_ids.device)
                self._mark_coord_tokens(token_mask, input_ids)
            else:
                token_mask = (input_ids == IMAGE_PAD_ID)  # [B, seq_len] bool
                # coord_routing: also route coordinate/bbox digit tokens to LoRA_V
                if self.coord_routing:
                    self._mark_coord_tokens(token_mask, input_ids)

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

        # Step 6: Balance loss (learned routing only)
        # Pushes the mean routing weight toward 0.5 per layer so neither
        # LoRA_WHERE nor LoRA_WHAT collapses to zero usage. Does NOT penalize
        # per-token confident routing (w near 0 or 1 is fine as long as the
        # average is balanced).
        L_balance = torch.tensor(0.0, device=L_act.device)
        mean_router_w = 0.0
        if self.routing_mode == "learned" and self.balance_weight > 0:
            balance_terms = []
            w_sum = 0.0
            w_count = 0
            for m in self.coop_modules:
                if m._last_router_w is None:
                    continue
                w = m._last_router_w  # [B, S, 1]
                mean_w = w.mean()
                # Binary entropy: -p log p - (1-p) log (1-p); maximized at p=0.5.
                # Minimize `-entropy` to push mean_w toward 0.5.
                eps = 1e-6
                neg_entropy = mean_w * torch.log(mean_w + eps) + (1 - mean_w) * torch.log(1 - mean_w + eps)
                balance_terms.append(neg_entropy)
                w_sum += mean_w.detach().item()
                w_count += 1
            if balance_terms:
                L_balance = torch.stack(balance_terms).mean()
                if w_count > 0:
                    mean_router_w = w_sum / w_count

        loss = L_act + self.bind_weight * L_bind + self.balance_weight * L_balance

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
        if self.routing_mode == "learned":
            diagnostics["L_balance"] = L_balance.detach() if isinstance(L_balance, torch.Tensor) else torch.tensor(L_balance)
            diagnostics["mean_router_w"] = mean_router_w

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

    # ── Inference mode override ─────────────────────────────────────

    def set_inference_mode(self, mode: str):
        """Override routing at inference. 'hard'|'v_only'|'t_only'|'merge'."""
        for m in self.coop_modules:
            m._inference_mode = mode
        print(f"[CooperativeVLMWrapper] inference mode = {mode}")

    # ── Router warm-start (v8) ──────────────────────────────────────

    @torch.no_grad()
    def warmstart_routers_from_token_type(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        pixel_values_list: List[torch.Tensor],
        image_grid_thw_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        where_bias: float = 2.2,   # sigmoid(2.2) ≈ 0.9 for WHERE class
        what_bias: float = -2.2,   # sigmoid(-2.2) ≈ 0.1 for WHAT class
    ):
        """Fit each layer's router from labeled hidden states (v8 warm-start).

        Labels per token (based on AC token structure):
          WHERE (w→1): IMAGE tokens + COORD digit tokens + THINK tokens
          WHAT  (w→0): ACTION tokens + OTHER response template tokens

        Strategy: collect hidden states per layer on a few samples, compute the
        mean direction (μ_WHERE - μ_WHAT), normalize, and set the router weight
        accordingly. This approximates a 1-step logistic regression but doesn't
        require sklearn / CPU round trip.

        Args:
            input_ids_list:     list of [seq_len] or [1, seq_len] tensors
            attention_mask_list: list of same shape
            pixel_values_list:  list of vision inputs (or None)
            image_grid_thw_list: list of [t,h,w] tensors
            labels_list:        list of [seq_len] label tensors (-100 = masked)
            where_bias, what_bias: sigmoid-logit targets for the two classes
        """
        if self.routing_mode != "learned" or self.routers is None:
            return

        device = next(self.base_model.parameters()).device
        layers = self._get_transformer_layers()
        num_layers = len(layers)
        hidden_size = self.routers[0].weight.shape[1]

        # Per-layer sums for WHERE / WHAT classes
        sum_where = [torch.zeros(hidden_size, device=device, dtype=torch.float32)
                     for _ in range(num_layers)]
        sum_what = [torch.zeros(hidden_size, device=device, dtype=torch.float32)
                    for _ in range(num_layers)]
        n_where = [0] * num_layers
        n_what = [0] * num_layers

        # Install forward hooks to capture hidden states out of each layer
        captured: Dict[int, torch.Tensor] = {}

        def make_hook(layer_idx):
            def hook_fn(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h.detach()
            return hook_fn

        handles = []
        for li, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(li)))

        try:
            # Disable router use during capture — base_model must see
            # routing_mode="learned" with initial zero router, which yields 0.5 blend.
            # For a faithful hidden state, we can either (a) run base only (bypass LoRA)
            # or (b) accept that initial hidden states are very close to base.
            # Simpler: bypass LoRA entirely by zeroing LoRA outputs via set_token_mask=None
            # and routing_mode temporarily set to a no-op. Since router is zero-init
            # both branches get equal 0.5 weight and delta=0.5*(δV+δA). With B=0 init
            # δV=δA=0, so there is no LoRA contribution anyway. We can just run forward.

            for ids, amask, pixel, grid, labels in zip(
                input_ids_list, attention_mask_list,
                pixel_values_list, image_grid_thw_list, labels_list,
            ):
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                    amask = amask.unsqueeze(0)
                    labels = labels.unsqueeze(0)
                ids = ids.to(device)
                amask = amask.to(device)
                labels = labels.to(device)
                kwargs = {}
                if pixel is not None:
                    kwargs["pixel_values"] = pixel.to(device)
                if grid is not None:
                    kwargs["image_grid_thw"] = grid.to(device).unsqueeze(0) if grid.dim() == 1 else grid.to(device)

                # Build per-token class labels:
                #   WHERE=1 for IMAGE tokens (task-independent) + assistant tokens
                #     that look like COORD or THINK content.
                #   WHAT=0 for ACTION tokens / other non-think/coord assistant tokens.
                #   Ignored for non-assistant tokens (labels == -100) except IMAGE.
                is_image = (ids == IMAGE_PAD_ID)
                # Find coord tokens via existing helper (reuse coord detection logic).
                coord_mask = torch.zeros_like(ids, dtype=torch.bool)
                self._mark_coord_tokens(coord_mask, ids)
                # Find think tokens via <think>/</think> trigram scan
                think_mask = self._find_think_spans(ids)

                # Assistant = positions where labels != -100.
                # (For simplicity, treat the full range of labels != -100 as assistant.)
                is_assistant = (labels != -100)

                # WHERE class: image (always) ∪ (assistant ∩ (coord ∨ think))
                where_class = is_image | (is_assistant & (coord_mask | think_mask))
                # WHAT class: assistant ∩ ¬where_class
                what_class = is_assistant & (~where_class)

                # Clear capture
                captured.clear()
                # Forward (no labels needed for hidden state collection)
                self.base_model(
                    input_ids=ids,
                    attention_mask=amask,
                    output_hidden_states=False,
                    return_dict=True,
                    **kwargs,
                )

                for li in range(num_layers):
                    if li not in captured:
                        continue
                    h = captured[li].float()  # [1, S, D]
                    # Aggregate
                    wh = h[where_class.unsqueeze(-1).expand_as(h)].view(-1, hidden_size)
                    wa = h[what_class.unsqueeze(-1).expand_as(h)].view(-1, hidden_size)
                    if wh.numel() > 0:
                        sum_where[li] += wh.sum(dim=0)
                        n_where[li] += wh.shape[0]
                    if wa.numel() > 0:
                        sum_what[li] += wa.sum(dim=0)
                        n_what[li] += wa.shape[0]
        finally:
            for h in handles:
                h.remove()

        # Compute per-layer routers: w = w_dir · h + b
        # where w_dir points WHERE→WHAT separation.
        print("[warmstart] Fitting routers from collected hidden states:")
        for li in range(num_layers):
            if n_where[li] == 0 or n_what[li] == 0:
                print(f"  layer {li}: insufficient samples, keeping zero init")
                continue
            mu_w = sum_where[li] / n_where[li]
            mu_a = sum_what[li] / n_what[li]
            direction = mu_w - mu_a
            dir_norm = direction.norm()
            if dir_norm < 1e-6:
                continue
            # Logistic regression closed-form approximation:
            # We want sigmoid(w·h + b) ≈ 0.9 at h = mu_w, ≈ 0.1 at h = mu_a
            # => w·mu_w + b =  where_bias,  w·mu_a + b = what_bias
            # If we set w = k * direction and b = -k * (mu_w + mu_a)/2 * direction + 0
            # ... but simpler: set slope so that:
            #   w·mu_w - w·mu_a = where_bias - what_bias
            #   k * ||direction||^2 = where_bias - what_bias
            #   k = (where_bias - what_bias) / ||direction||^2
            k = (where_bias - what_bias) / (dir_norm ** 2)
            w_vec = k * direction
            # bias: center at midpoint, so that midpoint maps to 0
            midpoint = 0.5 * (mu_w + mu_a)
            b = -(w_vec * midpoint).sum()
            router = self.routers[li]
            router.weight.data.copy_(w_vec.unsqueeze(0).to(router.weight.dtype))
            router.bias.data.fill_(float(b))
            # Report sanity
            pred_w = torch.sigmoid((w_vec * mu_w).sum() + b).item()
            pred_a = torch.sigmoid((w_vec * mu_a).sum() + b).item()
            print(f"  layer {li:2d}: n_WHERE={n_where[li]:6d} n_WHAT={n_what[li]:6d} "
                  f"sigmoid(mu_WHERE)={pred_w:.3f} sigmoid(mu_WHAT)={pred_a:.3f}")
        print("[warmstart] Done.")

    # ── Think span detection (for warm-start) ─────────────────────

    # <think> trigram: (13708, 766, 29); </think> trigram: (522, 26865, 29)
    _THINK_OPEN = (13708, 766, 29)
    _THINK_CLOSE = (522, 26865, 29)

    def _find_think_spans(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return [B, seq_len] bool mask where tokens lie inside <think>...</think>."""
        B, S = input_ids.shape
        mask = torch.zeros(B, S, dtype=torch.bool, device=input_ids.device)
        for b in range(B):
            ids = input_ids[b].tolist()
            in_think = False
            i = 0
            while i < S:
                if (i + 2 < S and
                        ids[i] == self._THINK_OPEN[0] and
                        ids[i + 1] == self._THINK_OPEN[1] and
                        ids[i + 2] == self._THINK_OPEN[2]):
                    in_think = True
                    mask[b, i] = True
                    mask[b, i + 1] = True
                    mask[b, i + 2] = True
                    i += 3
                    continue
                if (i + 2 < S and
                        ids[i] == self._THINK_CLOSE[0] and
                        ids[i + 1] == self._THINK_CLOSE[1] and
                        ids[i + 2] == self._THINK_CLOSE[2]):
                    mask[b, i] = True
                    mask[b, i + 1] = True
                    mask[b, i + 2] = True
                    in_think = False
                    i += 3
                    continue
                if in_think:
                    mask[b, i] = True
                i += 1
        return mask

    # ── Trainable parameters ────────────────────────────────────────

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return all trainable parameters (LoRA_V + LoRA_A + LoRA_T if 3-agent)."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Save / load ─────────────────────────────────────────────────

    def save_cooperative_checkpoint(self, output_dir: str):
        """Save adapter weights: lora_v.pt, lora_a.pt, (lora_t.pt), and lora_b_shared.pt when shared_B."""
        os.makedirs(output_dir, exist_ok=True)
        v_state, a_state, t_state, b_shared_state = {}, {}, {}, {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A_v" in name or "lora_B_v" in name:
                v_state[name] = param.data.clone().cpu()
            elif "lora_A_a" in name or "lora_B_a" in name:
                a_state[name] = param.data.clone().cpu()
            elif "lora_A_t" in name or "lora_B_t" in name:
                t_state[name] = param.data.clone().cpu()
            elif name.endswith(".lora_B"):
                # shared_B mode: single B per module
                b_shared_state[name] = param.data.clone().cpu()

        torch.save(v_state, os.path.join(output_dir, "lora_v.pt"))
        torch.save(a_state, os.path.join(output_dir, "lora_a.pt"))
        if t_state:
            torch.save(t_state, os.path.join(output_dir, "lora_t.pt"))
        if b_shared_state:
            torch.save(b_shared_state, os.path.join(output_dir, "lora_b_shared.pt"))

        # Save sep params for soft routing
        sep_state = {}
        if self.soft_routing:
            for name, param in self.named_parameters():
                if name.endswith(".sep") and param.requires_grad:
                    sep_state[name] = param.data.clone().cpu()
            if sep_state:
                torch.save(sep_state, os.path.join(output_dir, "lora_sep.pt"))

        # Save communication params (v6)
        comm_state = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ['W_av', 'W_va', 'gate_av', 'gate_va']):
                comm_state[name] = param.data.clone().cpu()
        if comm_state:
            torch.save(comm_state, os.path.join(output_dir, "lora_comm.pt"))

        # Save learned routers (v8)
        if self.routing_mode == "learned" and self.routers is not None:
            router_state = {}
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("routers."):
                    router_state[name] = param.data.clone().cpu()
            if router_state:
                torch.save(router_state, os.path.join(output_dir, "routers.pt"))

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
            "cooperative_comm": self.cooperative_comm,
            "gate_init": self.gate_init,
            "gate_type": self.gate_type,
            "routing_mode": self.routing_mode,
            "coord_only_routing": self.coord_only_routing,
            "balance_weight": self.balance_weight,
            "shared_B": self.shared_B,
        }
        if b_shared_state:
            config["lora_b_shared_params"] = sum(v.numel() for v in b_shared_state.values())
        if t_state:
            config["lora_t_params"] = sum(v.numel() for v in t_state.values())
        if sep_state:
            config["sep_values"] = {
                name: torch.sigmoid(param).item()
                for name, param in sep_state.items()
            }
        if comm_state:
            gate_values = {}
            act = torch.tanh if self.gate_type == "tanh" else torch.sigmoid
            for name, param in comm_state.items():
                # Match the scalar gate params, not the W_av/W_va matrices.
                # Note: target_modules may include "gate_proj", whose W_av/W_va
                # contain the substring "gate". Use endswith to be precise.
                if name.endswith(".gate_av") or name.endswith(".gate_va"):
                    gate_values[name] = round(act(param).item(), 6)
            config["gate_values"] = gate_values
            config["comm_params"] = sum(v.numel() for v in comm_state.values())
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

        # Load shared-B params (v8 shared_B mode)
        b_shared_path = os.path.join(checkpoint_dir, "lora_b_shared.pt")
        if os.path.exists(b_shared_path) and self.shared_B:
            b_shared_state = torch.load(b_shared_path, map_location="cpu", weights_only=True)
            loaded = 0
            for name, param in self.named_parameters():
                if name in b_shared_state:
                    param.data.copy_(b_shared_state[name].to(param.device))
                    loaded += 1
            print(f"Loaded {loaded} shared-B params from checkpoint")

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

        # Load communication params (v6)
        comm_path = os.path.join(checkpoint_dir, "lora_comm.pt")
        if os.path.exists(comm_path) and self.cooperative_comm:
            comm_state = torch.load(comm_path, map_location="cpu", weights_only=True)
            loaded = 0
            for name, param in self.named_parameters():
                if name in comm_state:
                    param.data.copy_(comm_state[name].to(param.device))
                    loaded += 1
            print(f"Loaded {loaded} communication params from checkpoint")

        # Load learned routers (v8)
        router_path = os.path.join(checkpoint_dir, "routers.pt")
        if os.path.exists(router_path) and self.routing_mode == "learned":
            router_state = torch.load(router_path, map_location="cpu", weights_only=True)
            loaded = 0
            for name, param in self.named_parameters():
                if name in router_state:
                    param.data.copy_(router_state[name].to(param.device))
                    loaded += 1
            print(f"Loaded {loaded} router params from checkpoint")
