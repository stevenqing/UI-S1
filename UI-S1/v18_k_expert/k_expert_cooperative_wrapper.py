"""V18 K-Expert Cooperative VLM Wrapper.

Extension of V13's IterativeCooperativeVLMWrapper from 2 experts to K experts
with softmax routing, configurable communication topology, and diversity loss.

Key changes from V13:
  - Route weights: [hidden_size] per layer → [hidden_size, K] per layer
  - Comm params: topology-dependent allocation (none, top2, shared, full)
  - Balance loss: K-way entropy regularization (was binary entropy)
  - Diversity loss: pairwise cosine similarity between expert outputs
  - Save format: lora_A.{i} keys, route matrices, topology config
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v18_k_expert.k_expert_cooperative_lora import KExpertCooperativeLoRALinear


class KExpertCooperativeVLMWrapper(nn.Module):
    """Wrap a Qwen2.5-VL model with K-expert cooperative LoRA."""

    def __init__(
        self,
        base_model: nn.Module,
        lora_r: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        balance_weight: float = 0.01,
        diversity_weight: float = 0.001,
        num_comm_rounds: int = 2,
        num_experts: int = 4,
        comm_topology: str = "top2",
    ):
        super().__init__()
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.balance_weight = balance_weight
        self.diversity_weight = diversity_weight
        self.num_comm_rounds = num_comm_rounds
        self.num_experts = num_experts
        self.comm_topology = comm_topology

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.target_modules = target_modules

        # Expose base model's config for HF Trainer compatibility
        self.config = getattr(base_model, "config", None)

        # Freeze ALL base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Get transformer layers info
        layers = self._get_transformer_layers()
        num_layers = len(layers)
        hidden_size = layers[0].self_attn.q_proj.in_features
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype

        # Per-layer routing matrices: [hidden_size, K]
        # Init zeros → softmax(0) = uniform 1/K
        self.route_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, num_experts, device=device, dtype=dtype))
            for _ in range(num_layers)
        ])

        # Per-layer communication parameters (topology-dependent)
        self._init_comm_params(num_layers, lora_r, device, dtype)

        # Replace target modules with KExpertCooperativeLoRALinear
        self.coop_modules: List[KExpertCooperativeLoRALinear] = []
        self._module_to_layer: List[int] = []
        self._replace_target_modules(lora_r, lora_alpha, lora_dropout, num_experts)

        # Attach communication params to each LoRA module
        self._attach_comm_params()

        print(f"[KExpertCooperativeVLMWrapper] {num_layers} layers, "
              f"{len(self.coop_modules)} replaced modules, "
              f"K={num_experts}, topology={comm_topology}, "
              f"target_modules={target_modules}, "
              f"num_comm_rounds={num_comm_rounds}")

    # -- Communication parameter initialization ----------------------------

    def _init_comm_params(self, num_layers: int, r: int, device, dtype):
        """Initialize communication parameters based on topology."""
        T = self.num_comm_rounds
        K = self.num_experts
        topology = self.comm_topology

        if topology == "none":
            # No communication parameters needed
            self.comm_W = nn.ParameterList()
            self.comm_gate = nn.ParameterList()
            return

        if topology in ("top2", "shared"):
            # Shared W_comm and gate across all pairs, per layer per round
            self.comm_W = nn.ParameterList([
                nn.Parameter(torch.zeros(r, r, device=device, dtype=dtype))
                for _ in range(num_layers * T)
            ])
            self.comm_gate = nn.ParameterList([
                nn.Parameter(torch.zeros(r, device=device, dtype=dtype))
                for _ in range(num_layers * T)
            ])

            # Init: W = kaiming_uniform, gate = zeros → sigmoid(0) = 0.5
            for i in range(num_layers * T):
                w_f = torch.zeros(r, r)
                nn.init.kaiming_uniform_(w_f, a=math.sqrt(5))
                self.comm_W[i].data.copy_(w_f.to(dtype))
                # gate stays zero

        elif topology == "full":
            # Each directed pair has its own W and gate
            num_pairs = K * (K - 1)
            self.comm_W = nn.ParameterList([
                nn.Parameter(torch.zeros(r, r, device=device, dtype=dtype))
                for _ in range(num_layers * T * num_pairs)
            ])
            self.comm_gate = nn.ParameterList([
                nn.Parameter(torch.zeros(r, device=device, dtype=dtype))
                for _ in range(num_layers * T * num_pairs)
            ])

            # Init
            for i in range(num_layers * T * num_pairs):
                w_f = torch.zeros(r, r)
                nn.init.kaiming_uniform_(w_f, a=math.sqrt(5))
                self.comm_W[i].data.copy_(w_f.to(dtype))

            # Build pair map: all (src, dst) where src != dst
            self._pair_map = []
            for src in range(K):
                for dst in range(K):
                    if src != dst:
                        self._pair_map.append((src, dst))
        else:
            raise ValueError(f"Unknown comm_topology: {topology}")

    # -- Module replacement ------------------------------------------------

    def _get_transformer_layers(self):
        """Locate the transformer layer ModuleList."""
        vlm = self.base_model.model
        if hasattr(vlm, "language_model"):
            return vlm.language_model.layers
        elif hasattr(vlm, "layers"):
            return vlm.layers
        else:
            raise AttributeError(
                f"Cannot find transformer layers in {type(vlm).__name__}. "
                f"Children: {[n for n, _ in vlm.named_children()]}"
            )

    def _replace_target_modules(self, r: int, alpha: int, dropout: float,
                                num_experts: int):
        """Replace nn.Linear with KExpertCooperativeLoRALinear."""
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
                coop_linear = KExpertCooperativeLoRALinear(
                    original, r, alpha, dropout, num_experts=num_experts,
                )
                # Attach shared per-layer routing matrix
                coop_linear.set_route_weight(self.route_weights[layer_idx])
                setattr(parent, module_name, coop_linear)
                self.coop_modules.append(coop_linear)
                self._module_to_layer.append(layer_idx)

    def _build_comm_dict(self, layer_idx: int) -> dict:
        """Build communication param dict for a specific layer."""
        T = self.num_comm_rounds
        K = self.num_experts
        topology = self.comm_topology

        if topology == "none":
            return {'T': T, 'topology': 'none'}

        if topology in ("top2", "shared"):
            base = layer_idx * T
            return {
                'T': T,
                'topology': topology,
                'W_comm': [self.comm_W[base + t] for t in range(T)],
                'gate': [self.comm_gate[base + t] for t in range(T)],
            }

        if topology == "full":
            num_pairs = K * (K - 1)
            base = layer_idx * T * num_pairs
            W_per_round = []
            gate_per_round = []
            for t in range(T):
                round_base = base + t * num_pairs
                W_per_round.append([self.comm_W[round_base + p] for p in range(num_pairs)])
                gate_per_round.append([self.comm_gate[round_base + p] for p in range(num_pairs)])
            return {
                'T': T,
                'topology': 'full',
                'W_comm': W_per_round,
                'gate': gate_per_round,
                'pair_map': self._pair_map,
            }

        raise ValueError(f"Unknown topology: {topology}")

    def _attach_comm_params(self):
        """Attach communication params to all LoRA modules."""
        for module, layer_idx in zip(self.coop_modules, self._module_to_layer):
            module.set_comm_params(self._build_comm_dict(layer_idx))

    # -- Gradient checkpointing --------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    # -- LoRA enable/disable (for ref model computation) -------------------

    def disable_lora(self):
        """Disable LoRA by clearing route weights and comm params."""
        for module in self.coop_modules:
            module.set_route_weight(None)
            module.set_comm_params(None)

    def enable_lora(self):
        """Re-enable LoRA by restoring route weights and comm params."""
        for module, layer_idx in zip(self.coop_modules, self._module_to_layer):
            module.set_route_weight(self.route_weights[layer_idx])
            module.set_comm_params(self._build_comm_dict(layer_idx))

    def enable_lora_layers(self, active_layers: list):
        """Enable LoRA only for specified layers, disable for all others."""
        active_set = set(active_layers)
        for module, layer_idx in zip(self.coop_modules, self._module_to_layer):
            if layer_idx in active_set:
                module.set_route_weight(self.route_weights[layer_idx])
                module.set_comm_params(self._build_comm_dict(layer_idx))
            else:
                module.set_route_weight(None)
                module.set_comm_params(None)

    # -- Expert output caching for diversity loss --------------------------

    def enable_expert_output_caching(self):
        """Enable caching of expert outputs for diversity loss."""
        for m in self.coop_modules:
            m._cache_expert_outputs = True

    def disable_expert_output_caching(self):
        """Disable caching and clean up."""
        for m in self.coop_modules:
            m._cache_expert_outputs = False
            m._last_expert_outputs = None

    # -- Routing noise for RL exploration ----------------------------------

    def set_routing_noise(self, std: float):
        """Set routing noise std for all modules (structured exploration)."""
        for module in self.coop_modules:
            module._routing_noise_std = std

    # -- Generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, input_ids=None, **kwargs):
        """Generate — routing and communication are automatic in forward."""
        return self.base_model.generate(input_ids=input_ids, **kwargs)

    # -- Forward (delegates to base model) ---------------------------------

    def forward(self, **kwargs):
        """Forward pass. Routing and communication happen inside each LoRA module."""
        return self.base_model(**kwargs)

    # -- Balance loss (K-way entropy regularization) -----------------------

    def compute_balance_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Routing entropy regularization — push toward uniform K-way routing.

        L = -entropy(p) = Σ_k p_k log(p_k)
        where p_k = mean routing weight for expert k across all tokens/modules.
        Minimizing L maximizes entropy → uniform routing.
        Perfect balance: entropy = log(K).

        Returns:
            loss: scalar tensor
            stats: dict with routing_entropy, per-expert usage, etc.
        """
        K = self.num_experts
        dev = next(self.parameters()).device

        if self.balance_weight <= 0:
            return torch.tensor(0.0, device=dev), {"routing_entropy": math.log(K)}

        # Collect mean routing weights across all modules
        all_weights = []
        for m in self.coop_modules:
            if m._last_routing_weights is not None:
                # [B, S, K] → mean across B, S → [K]
                all_weights.append(m._last_routing_weights.mean(dim=(0, 1)))

        if not all_weights:
            return torch.tensor(0.0, device=dev), {"routing_entropy": math.log(K)}

        # Average across modules → [K]
        p = torch.stack(all_weights).mean(dim=0)

        eps = 1e-8
        neg_entropy = (p * torch.log(p + eps)).sum()

        # Stats
        entropy = -neg_entropy.item()
        stats = {
            "routing_entropy": entropy,
            "max_expert_usage": p.max().item(),
            "min_expert_usage": p.min().item(),
        }
        for k in range(K):
            stats[f"expert_{k}_usage"] = p[k].item()

        return neg_entropy, stats

    # -- Diversity loss (new in V18) ---------------------------------------

    def compute_diversity_loss(self) -> torch.Tensor:
        """Mean pairwise cosine similarity between expert outputs.

        L = mean_{i<j} cos_sim(h_i_flat, h_j_flat)
        Minimizing L pushes expert representations apart.
        """
        K = self.num_experts
        dev = next(self.parameters()).device

        if self.diversity_weight <= 0:
            return torch.tensor(0.0, device=dev)

        cos_sims = []
        for module in self.coop_modules:
            eo = module._last_expert_outputs
            if eo is None or len(eo) < 2:
                continue

            # Stack and flatten: [K, B*S*r]
            eo_flat = torch.stack(eo, dim=0).reshape(K, -1)
            eo_norm = F.normalize(eo_flat, dim=-1)

            # Pairwise cosine similarity
            sim = eo_norm @ eo_norm.t()  # [K, K]

            # Upper triangular (i < j) — exclude diagonal
            mask = torch.triu(torch.ones(K, K, device=dev, dtype=torch.bool), diagonal=1)
            upper = sim[mask]
            if upper.numel() > 0:
                cos_sims.append(upper.mean())

        if not cos_sims:
            return torch.tensor(0.0, device=dev)

        return torch.stack(cos_sims).mean()

    # -- Trainable parameter count -----------------------------------------

    def count_trainable_params(self) -> Dict[str, int]:
        """Count trainable parameters by category."""
        lora_n = 0
        route_n = 0
        comm_n = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_n += param.numel()
            elif "comm_" in name:
                comm_n += param.numel()
            elif "lora_" in name:
                lora_n += param.numel()
        return {
            "lora": lora_n,
            "route_weights": route_n,
            "comm": comm_n,
            "total": lora_n + route_n + comm_n,
        }

    # -- Save/Load ---------------------------------------------------------

    def save_cooperative(self, save_dir: str):
        """Save cooperative LoRA weights, route weights, comm weights, and config."""
        os.makedirs(save_dir, exist_ok=True)

        lora_state = {}
        route_state = {}
        comm_state = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_state[name] = param.data.cpu()
            elif "comm_" in name:
                comm_state[name] = param.data.cpu()
            elif "lora_" in name:
                lora_state[name] = param.data.cpu()

        torch.save(lora_state, os.path.join(save_dir, "lora_weights.pt"))
        torch.save(route_state, os.path.join(save_dir, "route_weights.pt"))
        torch.save(comm_state, os.path.join(save_dir, "comm_weights.pt"))

        config = {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "balance_weight": self.balance_weight,
            "diversity_weight": self.diversity_weight,
            "num_comm_rounds": self.num_comm_rounds,
            "num_experts": self.num_experts,
            "comm_topology": self.comm_topology,
            "type": "k_expert_cooperative_v18",
        }
        with open(os.path.join(save_dir, "cooperative_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_cooperative(self, load_dir: str, device=None):
        """Load cooperative LoRA weights, route weights, and comm weights."""
        for fname in ("lora_weights.pt", "route_weights.pt", "comm_weights.pt"):
            fpath = os.path.join(load_dir, fname)
            if os.path.exists(fpath):
                state = torch.load(fpath, map_location=device or "cpu",
                                   weights_only=True)
                missing, unexpected = self.load_state_dict(state, strict=False)
                print(f"  Loaded {fname}: {len(state)} tensors "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
