from dataclasses import dataclass

import torch
from torch import nn


MAX_CANDIDATES = 12


class TriVUSSetRanker(nn.Module):
    def __init__(self, input_dim, width=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.keep_delta = nn.Parameter(torch.zeros(width))
        self.utility_head = nn.Linear(width, 1)
        self.fallback_correct_head = nn.Linear(width, 1)

    def forward(self, features, candidate_mask, fallback_indices):
        if features.ndim != 3 or features.shape[1] != MAX_CANDIDATES:
            raise ValueError("TriVUS feature shape mismatch")
        if candidate_mask.shape != features.shape[:2] or candidate_mask.dtype != torch.bool:
            raise ValueError("TriVUS candidate-mask shape mismatch")
        if fallback_indices.shape != (features.shape[0],):
            raise ValueError("TriVUS fallback shape mismatch")
        valid_counts = candidate_mask.sum(dim=1)
        if not torch.all((valid_counts == 3) | (valid_counts == 12)):
            raise ValueError("TriVUS valid candidate count must be 3 or 12")
        if not torch.isfinite(features).all():
            raise ValueError("TriVUS features must be finite")
        padding_features = features.masked_select(~candidate_mask[:, :, None])
        if torch.any(padding_features != 0):
            raise ValueError("TriVUS padding features must be zero")
        if torch.any(fallback_indices < 0) or torch.any(fallback_indices >= MAX_CANDIDATES):
            raise ValueError("TriVUS fallback index out of range")
        rows = torch.arange(features.shape[0], device=features.device)
        if not torch.all(candidate_mask[rows, fallback_indices]):
            raise ValueError("TriVUS fallback points to padding")
        candidates = self.candidate_encoder(features)
        keep = candidates[rows, fallback_indices] + self.keep_delta
        token_mask = torch.cat((
            candidate_mask,
            torch.ones((len(features), 1), dtype=torch.bool, device=features.device),
        ), dim=1)
        encoded = self.set_encoder(
            torch.cat((candidates, keep[:, None, :]), dim=1),
            src_key_padding_mask=~token_mask,
        )
        utility = self.utility_head(encoded).squeeze(-1)
        utility = utility.masked_fill(~token_mask, torch.finfo(utility.dtype).min)
        fallback_correct = self.fallback_correct_head(encoded[:, -1]).squeeze(-1)
        if not torch.isfinite(utility[token_mask]).all() or not torch.isfinite(fallback_correct).all():
            raise ValueError("TriVUS model produced non-finite valid logits")
        return utility, fallback_correct


@dataclass(frozen=True)
class TriVUSBatch:
    features: torch.Tensor
    candidate_mask: torch.Tensor
    fallback_indices: torch.Tensor
    target_distribution: torch.Tensor
    fallback_correct: torch.Tensor
    weights: torch.Tensor


def permute_batch(batch, permutations):
    if permutations.shape != batch.features.shape[:2]:
        raise ValueError("TriVUS permutation shape mismatch")
    feature_indices = permutations[:, :, None].expand_as(batch.features)
    inverse = torch.argsort(permutations, dim=1)
    rows = torch.arange(len(permutations), device=permutations.device)
    return TriVUSBatch(
        features=torch.gather(batch.features, 1, feature_indices),
        candidate_mask=torch.gather(batch.candidate_mask, 1, permutations),
        fallback_indices=inverse[rows, batch.fallback_indices],
        target_distribution=torch.cat((
            torch.gather(batch.target_distribution[:, :MAX_CANDIDATES], 1, permutations),
            batch.target_distribution[:, MAX_CANDIDATES:],
        ), dim=1),
        fallback_correct=batch.fallback_correct,
        weights=batch.weights,
    )


def restore_candidate_order(values, permutations):
    inverse = torch.argsort(permutations, dim=1)
    return torch.gather(values, 1, inverse)


def trivus_loss(model, batch, normalization=None):
    if batch.target_distribution.shape != (len(batch.features), MAX_CANDIDATES + 1):
        raise ValueError("TriVUS target shape mismatch")
    if not torch.isfinite(batch.target_distribution).all() or torch.any(batch.target_distribution < 0):
        raise ValueError("TriVUS target distribution must be finite and nonnegative")
    if not torch.allclose(
        batch.target_distribution.sum(dim=1),
        torch.ones(len(batch.features), device=batch.features.device),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("TriVUS target distribution must sum to one")
    invalid_mass = batch.target_distribution[:, :MAX_CANDIDATES].masked_select(~batch.candidate_mask)
    if torch.any(invalid_mass != 0):
        raise ValueError("TriVUS target assigns mass to padding")
    if batch.fallback_correct.shape != (len(batch.features),):
        raise ValueError("TriVUS fallback-correct shape mismatch")
    if not torch.isfinite(batch.fallback_correct).all() or not torch.all(
        (batch.fallback_correct == 0) | (batch.fallback_correct == 1)
    ):
        raise ValueError("TriVUS fallback-correct target must be binary")
    if batch.weights.shape != (len(batch.features),):
        raise ValueError("TriVUS weight shape mismatch")
    if not torch.isfinite(batch.weights).all() or torch.any(batch.weights < 0):
        raise ValueError("TriVUS weights must be finite and nonnegative")
    utility, fallback_logit = model(
        batch.features, batch.candidate_mask, batch.fallback_indices
    )
    log_probabilities = torch.log_softmax(utility, dim=-1)
    token_mask = torch.cat((
        batch.candidate_mask,
        torch.ones((len(batch.features), 1), dtype=torch.bool, device=batch.features.device),
    ), dim=1)
    safe_log_probabilities = torch.where(
        token_mask, log_probabilities, torch.zeros_like(log_probabilities)
    )
    listwise = -(batch.target_distribution * safe_log_probabilities).sum(dim=-1)
    auxiliary = nn.functional.binary_cross_entropy_with_logits(
        fallback_logit, batch.fallback_correct, reduction="none"
    )
    denominator = batch.weights.sum() if normalization is None else normalization
    denominator = denominator.clamp_min(torch.finfo(listwise.dtype).eps)
    loss = ((listwise + 0.5 * auxiliary) * batch.weights).sum() / denominator
    return loss, {
        "loss": float(loss.detach()),
        "listwise": float((listwise * batch.weights).sum().detach() / denominator),
        "auxiliary": float((auxiliary * batch.weights).sum().detach() / denominator),
    }