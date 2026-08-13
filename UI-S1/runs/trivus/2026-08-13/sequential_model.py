import math

import torch
from torch import nn


class SequentialCandidateVerifier(nn.Module):
    def __init__(self, input_dimension, width=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        if input_dimension < 1 or width < 1 or heads < 1 or layers < 1:
            raise ValueError("Sequential verifier dimensions must be positive")
        self.input_dimension = int(input_dimension)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dimension, width),
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
        self.success_head = nn.Linear(width, 1)

    def forward(self, features, candidate_mask):
        if (
            features.ndim != 3
            or features.shape[-1] != self.input_dimension
            or candidate_mask.shape != features.shape[:2]
            or candidate_mask.dtype != torch.bool
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Sequential verifier input mismatch")
        counts = candidate_mask.sum(dim=1)
        if not torch.all((counts == 3) | (counts == 12)):
            raise ValueError("Sequential verifier requires 3 or 12 candidates")
        if torch.any(features.masked_select(~candidate_mask[:, :, None]) != 0):
            raise ValueError("Sequential verifier padding must be zero")
        encoded = self.set_encoder(
            self.candidate_encoder(features),
            src_key_padding_mask=~candidate_mask,
        )
        logits = self.success_head(encoded).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, 0.0)
        if not torch.isfinite(logits).all():
            raise ValueError("Sequential verifier produced non-finite logits")
        return logits, encoded


def cheap_oof_features(logits, candidate_mask, fallback_indices):
    if (
        logits.ndim != 2
        or candidate_mask.shape != logits.shape
        or candidate_mask.dtype != torch.bool
        or fallback_indices.shape != (len(logits),)
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("Cheap OOF feature input mismatch")
    rows = torch.arange(len(logits), device=logits.device)
    if (
        torch.any(fallback_indices < 0)
        or torch.any(fallback_indices >= logits.shape[1])
        or not torch.all(candidate_mask[rows, fallback_indices])
    ):
        raise ValueError("Cheap OOF fallback mismatch")
    masked = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
    probabilities = torch.sigmoid(masked)
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    inverse = torch.argsort(order, dim=1)
    counts = candidate_mask.sum(dim=1).to(logits.dtype)
    normalized_rank = inverse.to(logits.dtype) / (counts[:, None] - 1).clamp_min(1)
    valid_probabilities = probabilities.masked_fill(~candidate_mask, 0.0)
    normalized = valid_probabilities / valid_probabilities.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).eps
    )
    entropy = -(normalized * torch.log(normalized.clamp_min(1e-12))).sum(dim=1)
    entropy = entropy / torch.log(counts)
    top = torch.gather(masked, 1, order[:, :2])
    margin = top[:, 0] - top[:, 1]
    fallback_probability = probabilities[rows, fallback_indices]
    extras = torch.stack((
        probabilities,
        normalized_rank,
        entropy[:, None].expand_as(logits),
        margin[:, None].expand_as(logits),
        fallback_probability[:, None].expand_as(logits),
    ), dim=-1)
    extras = extras.masked_fill(~candidate_mask[:, :, None], 0.0)
    if extras.shape != (*logits.shape, 5) or not torch.isfinite(extras).all():
        raise ValueError("Cheap OOF features are invalid")
    return extras, order


def augment_verifier_features(base_features, cheap_features, candidate_mask):
    if (
        base_features.ndim != 3
        or base_features.shape[-1] != 115
        or cheap_features.shape != (*base_features.shape[:2], 5)
        or candidate_mask.shape != base_features.shape[:2]
        or candidate_mask.dtype != torch.bool
        or not torch.isfinite(base_features).all()
        or not torch.isfinite(cheap_features).all()
    ):
        raise ValueError("Sequential verifier feature augmentation mismatch")
    output = torch.cat((base_features, cheap_features), dim=-1)
    output = output.masked_fill(~candidate_mask[:, :, None], 0.0)
    return output