from dataclasses import dataclass

import torch
from torch import nn


CONFIGS = {
    "S1": {"learning_rate": 3e-4, "weight_decay": 1e-3, "epochs": 30, "aux_weight": 0.0, "grpo_weight": 0.0},
    "S2": {"learning_rate": 3e-4, "weight_decay": 1e-3, "epochs": 30, "aux_weight": 0.5, "grpo_weight": 0.0},
    "S3": {"learning_rate": 1e-4, "weight_decay": 1e-3, "epochs": 50, "aux_weight": 0.5, "grpo_weight": 0.25},
}


class VisualLogitSetRanker(nn.Module):
    def __init__(self, input_dim, width=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.keep_delta = nn.Parameter(torch.zeros(width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.utility_head = nn.Linear(width, 1)
        self.fallback_correct_head = nn.Linear(width, 1)

    def forward(self, features, fallback_indices):
        candidates = self.candidate_encoder(features)
        batch = torch.arange(candidates.shape[0], device=candidates.device)
        keep = candidates[batch, fallback_indices] + self.keep_delta
        encoded = self.set_encoder(torch.cat((candidates, keep[:, None, :]), dim=1))
        utility_logits = self.utility_head(encoded).squeeze(-1)
        fallback_correct_logits = self.fallback_correct_head(encoded[:, -1]).squeeze(-1)
        return utility_logits, fallback_correct_logits


@dataclass(frozen=True)
class RankerBatch:
    features: torch.Tensor
    fallback_indices: torch.Tensor
    target_distribution: torch.Tensor
    fallback_correct: torch.Tensor
    grpo_advantage: torch.Tensor
    weights: torch.Tensor


def weighted_mean(values, weights):
    denominator = weights.sum().clamp_min(torch.finfo(values.dtype).eps)
    return (values * weights).sum() / denominator


def ranker_loss(model, batch, config_id, normalization=None):
    config = CONFIGS[config_id]
    utility_logits, fallback_correct_logits = model(batch.features, batch.fallback_indices)
    log_probabilities = torch.log_softmax(utility_logits, dim=-1)
    listwise = -(batch.target_distribution * log_probabilities).sum(dim=-1)
    auxiliary = nn.functional.binary_cross_entropy_with_logits(
        fallback_correct_logits, batch.fallback_correct, reduction="none"
    )
    expected_advantage = -(torch.softmax(utility_logits, dim=-1) * batch.grpo_advantage).sum(dim=-1)
    per_row = listwise + config["aux_weight"] * auxiliary + config["grpo_weight"] * expected_advantage
    loss = (
        weighted_mean(per_row, batch.weights)
        if normalization is None
        else (per_row * batch.weights).sum() / normalization
    )
    metrics = {
        "loss": float(loss.detach()),
        "listwise": float(weighted_mean(listwise, batch.weights).detach()),
        "auxiliary": float(weighted_mean(auxiliary, batch.weights).detach()),
        "negative_expected_advantage": float(weighted_mean(expected_advantage, batch.weights).detach()),
    }
    return loss, metrics


def permute_batch(batch, permutations):
    if permutations.shape != batch.features.shape[:2]:
        raise ValueError("permutation shape mismatch")
    gather = permutations[:, :, None].expand_as(batch.features)
    features = torch.gather(batch.features, 1, gather)
    targets = torch.gather(batch.target_distribution[:, :12], 1, permutations)
    advantages = torch.gather(batch.grpo_advantage[:, :12], 1, permutations)
    inverse = torch.argsort(permutations, dim=1)
    batch_indices = torch.arange(len(permutations), device=permutations.device)
    fallback = inverse[batch_indices, batch.fallback_indices]
    return RankerBatch(
        features=features,
        fallback_indices=fallback,
        target_distribution=torch.cat((targets, batch.target_distribution[:, 12:]), dim=1),
        fallback_correct=batch.fallback_correct,
        grpo_advantage=torch.cat((advantages, batch.grpo_advantage[:, 12:]), dim=1),
        weights=batch.weights,
    )
