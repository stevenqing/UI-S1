from dataclasses import dataclass

import torch
from torch import nn


class AcquisitionRouter(nn.Module):
    def __init__(self, input_dim, width=64, layers=2, heads=4, dropout=0.1):
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
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.arm_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))

    def forward(self, features):
        encoded = self.set_encoder(self.candidate_encoder(features))
        return self.arm_head(encoded.mean(dim=1))


@dataclass(frozen=True)
class RouterBatch:
    features: torch.Tensor
    targets: torch.Tensor
    target_distribution: torch.Tensor
    listwise_active: torch.Tensor
    weights: torch.Tensor


def router_loss(model, batch, normalization=None):
    logits = model(batch.features)
    listwise = -(batch.target_distribution * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    listwise = listwise * batch.listwise_active
    binary = nn.functional.binary_cross_entropy_with_logits(logits, batch.targets, reduction="none").mean(dim=-1)
    per_row = listwise + 0.25 * binary
    denominator = batch.weights.sum() if normalization is None else normalization
    loss = (per_row * batch.weights).sum() / denominator.clamp_min(torch.finfo(per_row.dtype).eps)
    return loss, {
        "loss": float(loss.detach()),
        "listwise": float((listwise * batch.weights).sum().detach() / denominator),
        "binary": float((binary * batch.weights).sum().detach() / denominator),
    }


def permute_router_batch(batch, permutations):
    if permutations.shape != batch.features.shape[:2]:
        raise ValueError("router permutation shape mismatch")
    gather = permutations[:, :, None].expand_as(batch.features)
    return RouterBatch(
        features=torch.gather(batch.features, 1, gather),
        targets=batch.targets,
        target_distribution=batch.target_distribution,
        listwise_active=batch.listwise_active,
        weights=batch.weights,
    )
