import torch
from torch import nn


TIE = 0
WIN = 1
LOSS = 2


class IncrementalUtilityHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("Incremental utility width must be positive")
        self.projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )

    def forward(self, representation):
        if representation.ndim != 2:
            raise ValueError("Incremental utility representation must be rank two")
        if not torch.isfinite(representation).all():
            raise ValueError("Incremental utility representation must be finite")
        logits = self.projection(representation)
        if not torch.isfinite(logits).all():
            raise ValueError("Incremental utility logits must be finite")
        return logits


def incremental_labels(direct_success, baseline_success):
    if (
        direct_success.dtype != torch.bool
        or baseline_success.dtype != torch.bool
        or direct_success.shape != baseline_success.shape
        or direct_success.ndim != 1
    ):
        raise ValueError("Incremental utility outcomes must be aligned boolean vectors")
    labels = torch.full_like(direct_success, TIE, dtype=torch.long)
    labels[direct_success & ~baseline_success] = WIN
    labels[~direct_success & baseline_success] = LOSS
    return labels


def incremental_utility_loss(logits, labels, weights=None):
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("Incremental utility logits must have three classes")
    if labels.shape != (len(logits),) or labels.dtype != torch.long:
        raise ValueError("Incremental utility labels mismatch")
    if torch.any(labels < TIE) or torch.any(labels > LOSS):
        raise ValueError("Incremental utility label out of range")
    losses = nn.functional.cross_entropy(logits, labels, reduction="none")
    if weights is None:
        return losses.mean()
    if (
        weights.shape != labels.shape
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0)
        or not bool(weights.sum() > 0)
    ):
        raise ValueError("Incremental utility weights mismatch")
    return (losses * weights).sum() / weights.sum()


def incremental_scores(logits):
    if logits.ndim != 2 or logits.shape[1] != 3 or not torch.isfinite(logits).all():
        raise ValueError("Incremental utility logits mismatch")
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities[:, WIN] - probabilities[:, LOSS], probabilities[:, LOSS]


def apply_incremental_gate(
    direct_success,
    baseline_success,
    expected_delta,
    loss_probability,
    minimum_delta,
    maximum_loss_probability,
):
    if (
        direct_success.dtype != torch.bool
        or baseline_success.dtype != torch.bool
        or direct_success.shape != baseline_success.shape
        or expected_delta.shape != direct_success.shape
        or loss_probability.shape != direct_success.shape
        or not torch.isfinite(expected_delta).all()
        or not torch.isfinite(loss_probability).all()
        or torch.any(loss_probability < 0)
        or torch.any(loss_probability > 1)
    ):
        raise ValueError("Incremental gate input mismatch")
    if minimum_delta < 0 or not 0 <= maximum_loss_probability <= 1:
        raise ValueError("Incremental gate threshold mismatch")
    override = (
        (expected_delta >= minimum_delta)
        & (loss_probability <= maximum_loss_probability)
    )
    output = torch.where(override, direct_success, baseline_success)
    return output, override