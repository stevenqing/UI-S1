from dataclasses import dataclass

import torch
from torch import nn


CHANNELS = ("vus_binding", "global_semantic", "fine_local", "context_local", "random_placebo")
VARIANT_CHANNELS = {
    "FULL": ("vus_binding", "global_semantic", "fine_local", "context_local"),
    "VUS_ONLY": ("vus_binding",),
    "VUS_GLOBAL": ("vus_binding", "global_semantic"),
    "VUS_LOCAL": ("vus_binding", "fine_local", "context_local"),
    "RANDOM_PLACEBO": ("vus_binding", "global_semantic", "random_placebo"),
}


def channel_mask(variant, device=None):
    if variant not in VARIANT_CHANNELS:
        raise ValueError(f"unknown DELTA variant: {variant}")
    active = set(VARIANT_CHANNELS[variant])
    return torch.tensor([name in active for name in CHANNELS], dtype=torch.bool, device=device)


class DeltaLateFusion(nn.Module):
    def __init__(
        self,
        base_dim,
        channel_dim=7,
        channel_width=32,
        gate_width=32,
        candidate_width=64,
        layers=2,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Linear(base_dim, channel_width), nn.GELU(), nn.LayerNorm(channel_width)
        )
        self.channel_encoder = nn.Sequential(
            nn.Linear(channel_dim, channel_width), nn.GELU(), nn.LayerNorm(channel_width)
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(channel_width * 2, gate_width), nn.GELU(), nn.Linear(gate_width, 1)
        )
        self.candidate_fusion = nn.Sequential(
            nn.Linear(channel_width * 2, candidate_width), nn.GELU(), nn.LayerNorm(candidate_width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=candidate_width,
            nhead=heads,
            dim_feedforward=candidate_width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.keep_delta = nn.Parameter(torch.zeros(candidate_width))
        self.utility_head = nn.Linear(candidate_width, 1)
        self.fallback_correct_head = nn.Linear(candidate_width, 1)

    def forward(self, base_features, channel_features, fallback_indices, active_channels):
        if active_channels.shape != (len(CHANNELS),) or not active_channels.any():
            raise ValueError("invalid DELTA active-channel mask")
        base = self.base_encoder(base_features)
        channels = self.channel_encoder(channel_features)
        repeated_base = base[:, :, None, :].expand(-1, -1, len(CHANNELS), -1)
        gate_input = torch.cat((repeated_base, channels), dim=-1)
        gate_logits = self.channel_gate(gate_input).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~active_channels[None, None, :], float("-inf"))
        gate_probabilities = torch.softmax(gate_logits, dim=-1)
        fused_channel = (gate_probabilities[..., None] * channels).sum(dim=2)
        candidates = self.candidate_fusion(torch.cat((base, fused_channel), dim=-1))
        batch = torch.arange(candidates.shape[0], device=candidates.device)
        keep = candidates[batch, fallback_indices] + self.keep_delta
        encoded = self.set_encoder(torch.cat((candidates, keep[:, None, :]), dim=1))
        utility_logits = self.utility_head(encoded).squeeze(-1)
        fallback_correct_logits = self.fallback_correct_head(encoded[:, -1]).squeeze(-1)
        return utility_logits, fallback_correct_logits, gate_probabilities


@dataclass(frozen=True)
class DeltaBatch:
    base_features: torch.Tensor
    channel_features: torch.Tensor
    fallback_indices: torch.Tensor
    target_distribution: torch.Tensor
    fallback_correct: torch.Tensor
    grpo_advantage: torch.Tensor
    weights: torch.Tensor


def permute_batch(batch, permutations):
    if permutations.shape != batch.base_features.shape[:2]:
        raise ValueError("DELTA permutation shape mismatch")
    base_gather = permutations[:, :, None].expand_as(batch.base_features)
    channel_gather = permutations[:, :, None, None].expand_as(batch.channel_features)
    inverse = torch.argsort(permutations, dim=1)
    rows = torch.arange(len(permutations), device=permutations.device)
    fallback = inverse[rows, batch.fallback_indices]
    return DeltaBatch(
        base_features=torch.gather(batch.base_features, 1, base_gather),
        channel_features=torch.gather(batch.channel_features, 1, channel_gather),
        fallback_indices=fallback,
        target_distribution=torch.cat((
            torch.gather(batch.target_distribution[:, :12], 1, permutations),
            batch.target_distribution[:, 12:],
        ), dim=1),
        fallback_correct=batch.fallback_correct,
        grpo_advantage=torch.cat((
            torch.gather(batch.grpo_advantage[:, :12], 1, permutations),
            batch.grpo_advantage[:, 12:],
        ), dim=1),
        weights=batch.weights,
    )


def restore_candidate_order(values, permutations):
    inverse = torch.argsort(permutations, dim=1)
    if values.ndim == 2:
        return torch.gather(values, 1, inverse)
    if values.ndim == 3:
        return torch.gather(values, 1, inverse[:, :, None].expand_as(values))
    raise ValueError("unsupported DELTA restore rank")


def delta_loss(model, batch, variant, consistency_permutations, normalization=None):
    active = channel_mask(variant, batch.base_features.device)
    utility, fallback_logit, gate = model(
        batch.base_features, batch.channel_features, batch.fallback_indices, active
    )
    log_probabilities = torch.log_softmax(utility, dim=-1)
    listwise = -(batch.target_distribution * log_probabilities).sum(dim=-1)
    auxiliary = nn.functional.binary_cross_entropy_with_logits(
        fallback_logit, batch.fallback_correct, reduction="none"
    )
    expected_advantage = -(torch.softmax(utility, dim=-1) * batch.grpo_advantage).sum(dim=-1)

    changed = permute_batch(batch, consistency_permutations)
    changed_utility, changed_fallback, changed_gate = model(
        changed.base_features, changed.channel_features, changed.fallback_indices, active
    )
    restored_candidates = restore_candidate_order(changed_utility[:, :12], consistency_permutations)
    restored_gate = restore_candidate_order(changed_gate, consistency_permutations)
    consistency = (
        nn.functional.mse_loss(utility[:, :12], restored_candidates, reduction="none").mean(dim=1)
        + nn.functional.mse_loss(utility[:, 12], changed_utility[:, 12], reduction="none")
        + nn.functional.mse_loss(fallback_logit, changed_fallback, reduction="none")
        + nn.functional.mse_loss(gate, restored_gate, reduction="none").mean(dim=(1, 2))
    )
    per_row = listwise + 0.5 * auxiliary + 0.1 * consistency + 0.1 * expected_advantage
    denominator = batch.weights.sum() if normalization is None else normalization
    denominator = denominator.clamp_min(torch.finfo(per_row.dtype).eps)
    loss = (per_row * batch.weights).sum() / denominator
    return loss, {
        "loss": float(loss.detach()),
        "listwise": float((listwise * batch.weights).sum().detach() / denominator),
        "auxiliary": float((auxiliary * batch.weights).sum().detach() / denominator),
        "consistency": float((consistency * batch.weights).sum().detach() / denominator),
        "negative_expected_advantage": float((expected_advantage * batch.weights).sum().detach() / denominator),
    }
