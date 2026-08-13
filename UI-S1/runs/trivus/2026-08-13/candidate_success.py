import torch
from torch import nn


class CandidateSuccessHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError("Candidate success width must be positive")
        self.projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, candidate_representations, candidate_mask):
        if (
            candidate_representations.ndim != 3
            or candidate_mask.shape != candidate_representations.shape[:2]
            or candidate_mask.dtype != torch.bool
            or not torch.isfinite(candidate_representations).all()
            or torch.any(candidate_mask.sum(dim=1) < 1)
        ):
            raise ValueError("Candidate success representation mismatch")
        logits = self.projection(candidate_representations).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, 0.0)
        if not torch.isfinite(logits).all():
            raise ValueError("Candidate success logits must be finite")
        return logits


def candidate_success_loss(
    logits,
    labels,
    candidate_mask,
    row_weights=None,
    pairwise_weight=0.5,
):
    if (
        logits.ndim != 2
        or labels.shape != logits.shape
        or candidate_mask.shape != logits.shape
        or labels.dtype != torch.bool
        or candidate_mask.dtype != torch.bool
        or not torch.isfinite(logits).all()
        or torch.any(candidate_mask.sum(dim=1) < 1)
    ):
        raise ValueError("Candidate success loss input mismatch")
    if not isinstance(pairwise_weight, (int, float)) or pairwise_weight < 0:
        raise ValueError("Candidate success pairwise weight mismatch")
    binary = nn.functional.binary_cross_entropy_with_logits(
        logits, labels.to(logits.dtype), reduction="none"
    )
    valid_count = candidate_mask.sum(dim=1).to(logits.dtype)
    row_bce = (binary * candidate_mask).sum(dim=1) / valid_count
    pairwise_rows = []
    pairwise_active = []
    for row in range(len(logits)):
        positive = logits[row][candidate_mask[row] & labels[row]]
        negative = logits[row][candidate_mask[row] & ~labels[row]]
        if len(positive) and len(negative):
            differences = positive[:, None] - negative[None, :]
            pairwise_rows.append(nn.functional.softplus(-differences).mean())
            pairwise_active.append(True)
        else:
            pairwise_rows.append(logits[row].sum() * 0.0)
            pairwise_active.append(False)
    row_pairwise = torch.stack(pairwise_rows)
    if row_weights is None:
        weights = torch.ones_like(row_bce)
    else:
        if (
            row_weights.shape != row_bce.shape
            or not torch.isfinite(row_weights).all()
            or torch.any(row_weights < 0)
            or not bool(row_weights.sum() > 0)
        ):
            raise ValueError("Candidate success row weights mismatch")
        weights = row_weights.to(row_bce.dtype)
    denominator = weights.sum()
    bce_loss = (row_bce * weights).sum() / denominator
    pairwise_loss = (row_pairwise * weights).sum() / denominator
    loss = bce_loss + float(pairwise_weight) * pairwise_loss
    return loss, {
        "bce": bce_loss.detach(),
        "pairwise": pairwise_loss.detach(),
        "pairwise_active_rows": sum(pairwise_active),
    }


def select_candidate(logits, candidate_mask):
    if (
        logits.ndim != 2
        or candidate_mask.shape != logits.shape
        or candidate_mask.dtype != torch.bool
        or not torch.isfinite(logits).all()
        or torch.any(candidate_mask.sum(dim=1) < 1)
    ):
        raise ValueError("Candidate selection input mismatch")
    masked = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
    return torch.argmax(masked, dim=1)