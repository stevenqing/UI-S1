# Copyright 2024 UI-S1 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MoE Loss Functions for GUI Agent.

This module implements loss functions for training MoE routing:
1. Load Balance Loss: Ensures balanced expert utilization
2. Router Z-Loss: Regularizes router logits for training stability
3. Combined MoE Loss: Total loss for training

Design Decisions:
- MSE-based balance loss: Simple and effective for small number of experts
- Optional Z-loss: Prevents router logits from growing too large
- Configurable weights: Allow tuning balance vs task performance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoELossOutput:
    """Output from MoE loss computation."""

    total_loss: torch.Tensor
    lm_loss: torch.Tensor
    balance_loss: torch.Tensor
    z_loss: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for logging."""
        result = {
            'total_loss': self.total_loss.item(),
            'lm_loss': self.lm_loss.item(),
            'balance_loss': self.balance_loss.item(),
        }
        if self.z_loss is not None:
            result['z_loss'] = self.z_loss.item()
        return result


class LoadBalanceLoss(nn.Module):
    """
    Load Balance Loss for MoE routing.

    Encourages balanced expert utilization by penalizing deviation from
    uniform distribution across experts.

    Supported types:
    - 'mse': MSE between actual and target (uniform) distribution
    - 'switch': Switch Transformer style auxiliary loss
    - 'entropy': Maximize routing entropy (uniform = high entropy)

    Args:
        num_experts: Number of experts
        balance_type: Type of balance loss ('mse', 'switch', 'entropy')
        target_distribution: Target expert distribution (default: uniform)

    Reference:
        - Switch Transformer: https://arxiv.org/abs/2101.03961
        - GShard: https://arxiv.org/abs/2006.16668
    """

    VALID_TYPES = {'mse', 'switch', 'entropy'}

    def __init__(
        self,
        num_experts: int,
        balance_type: str = 'mse',
        target_distribution: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if balance_type not in self.VALID_TYPES:
            raise ValueError(
                f"balance_type must be one of {self.VALID_TYPES}, got {balance_type}"
            )

        self.num_experts = num_experts
        self.balance_type = balance_type

        # Target distribution (default: uniform)
        if target_distribution is None:
            target_distribution = torch.ones(num_experts) / num_experts
        self.register_buffer('target_distribution', target_distribution)

    def forward(
        self,
        routing_weights: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute load balance loss.

        Args:
            routing_weights: [B, num_experts] Softmax routing probabilities
            router_logits: [B, num_experts] Raw logits (needed for some loss types)

        Returns:
            balance_loss: Scalar loss value
        """
        if self.balance_type == 'mse':
            return self._mse_loss(routing_weights)
        elif self.balance_type == 'switch':
            return self._switch_loss(routing_weights)
        elif self.balance_type == 'entropy':
            return self._entropy_loss(routing_weights)
        else:
            raise ValueError(f"Unknown balance_type: {self.balance_type}")

    def _mse_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        MSE-based balance loss.

        Computes MSE between actual expert distribution and target (uniform).
        Simple and effective for small number of experts.
        """
        # Compute actual distribution: average routing weights over batch
        # actual_dist: [num_experts] - average probability per expert
        actual_dist = routing_weights.mean(dim=0)

        # MSE between actual and target
        target = self.target_distribution.to(routing_weights.device)
        loss = F.mse_loss(actual_dist, target)

        return loss

    def _switch_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer auxiliary load balancing loss.

        L_aux = num_experts * sum_i(f_i * P_i)

        Where:
        - f_i = fraction of tokens routed to expert i (batch avg of argmax)
        - P_i = fraction of routing probability to expert i (batch avg of soft)

        This loss penalizes both concentrated routing and uneven probabilities.
        """
        batch_size = routing_weights.size(0)

        # f_i: fraction of tokens routed to each expert (hard assignment)
        # Get dominant expert for each sample
        dominant_experts = routing_weights.argmax(dim=-1)  # [B]

        # Count tokens per expert
        expert_counts = torch.zeros(
            self.num_experts, device=routing_weights.device
        )
        for i in range(self.num_experts):
            expert_counts[i] = (dominant_experts == i).float().sum()

        # f_i = count / batch_size
        f = expert_counts / batch_size  # [num_experts]

        # P_i: average routing probability per expert
        P = routing_weights.mean(dim=0)  # [num_experts]

        # Switch loss: num_experts * sum(f_i * P_i)
        loss = self.num_experts * (f * P).sum()

        return loss

    def _entropy_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Entropy-based balance loss.

        Maximizes entropy of the average routing distribution.
        Higher entropy = more uniform distribution.

        Loss = -H(avg_routing) = -sum(p * log(p))

        Note: Returns negative because we want to maximize entropy.
        """
        # Average distribution
        avg_dist = routing_weights.mean(dim=0)  # [num_experts]

        # Entropy (negate because we minimize loss but want max entropy)
        log_dist = torch.log(avg_dist + 1e-10)
        entropy = -(avg_dist * log_dist).sum()

        # Max entropy for uniform distribution = log(num_experts)
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float))

        # Loss = 1 - normalized_entropy (so loss=0 when uniform)
        normalized_entropy = entropy / max_entropy.to(routing_weights.device)
        loss = 1.0 - normalized_entropy

        return loss


class RouterZLoss(nn.Module):
    """
    Router Z-Loss for training stability.

    Regularizes router logits to prevent them from growing too large,
    which can cause training instability with softmax.

    Z-Loss = mean(log(sum(exp(logits))))

    Reference: ST-MoE (https://arxiv.org/abs/2202.08906)

    Args:
        z_loss_weight: Weight for Z-loss (typical: 1e-3 to 1e-2)
    """

    def __init__(self, z_loss_weight: float = 0.001):
        super().__init__()
        self.z_loss_weight = z_loss_weight

    def forward(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Z-loss.

        Args:
            router_logits: [B, num_experts] Raw router logits

        Returns:
            z_loss: Scalar loss value
        """
        # ST-MoE z-loss: mean(logsumexp(logits)^2)
        # The square ensures the loss penalizes large logit magnitudes
        # regardless of sign (prevents logits drifting to large negatives).
        log_z = torch.logsumexp(router_logits, dim=-1)  # [B]
        z_loss = (log_z ** 2).mean()

        return z_loss * self.z_loss_weight


class MoELoss(nn.Module):
    """
    Combined MoE Loss for training.

    Combines:
    1. LM loss (cross-entropy for language modeling)
    2. Load balance loss (encourages uniform expert usage)
    3. Optional Z-loss (training stability)

    Total = LM_loss + balance_weight * balance_loss + z_weight * z_loss

    Args:
        num_experts: Number of experts
        balance_weight: Weight for balance loss (typical: 0.01 to 0.1)
        balance_type: Type of balance loss ('mse', 'switch', 'entropy')
        z_loss_weight: Weight for Z-loss (0 to disable)

    Example:
        >>> moe_loss = MoELoss(num_experts=4, balance_weight=0.1)
        >>> output = moe_loss(
        ...     lm_loss=ce_loss,
        ...     routing_weights=routing_probs,
        ...     router_logits=logits,
        ... )
        >>> output.total_loss.backward()
    """

    def __init__(
        self,
        num_experts: int = 4,
        balance_weight: float = 0.1,
        balance_type: str = 'mse',
        z_loss_weight: float = 0.0,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.balance_weight = balance_weight
        self.z_loss_weight = z_loss_weight

        # Balance loss
        self.balance_loss_fn = LoadBalanceLoss(
            num_experts=num_experts,
            balance_type=balance_type,
        )

        # Z-loss (optional)
        self.z_loss_fn = RouterZLoss(z_loss_weight=1.0) if z_loss_weight > 0 else None

    def forward(
        self,
        lm_loss: torch.Tensor,
        routing_weights: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> MoELossOutput:
        """
        Compute combined MoE loss.

        Args:
            lm_loss: Scalar language modeling loss
            routing_weights: [B, num_experts] Softmax routing probabilities
            router_logits: [B, num_experts] Raw logits (optional, for Z-loss)

        Returns:
            MoELossOutput with total_loss, lm_loss, balance_loss, z_loss
        """
        # Balance loss
        balance_loss = self.balance_loss_fn(routing_weights, router_logits)

        # Z-loss (optional)
        z_loss = None
        if self.z_loss_fn is not None and router_logits is not None:
            z_loss = self.z_loss_fn(router_logits) * self.z_loss_weight

        # Total loss
        total_loss = lm_loss + self.balance_weight * balance_loss
        if z_loss is not None:
            total_loss = total_loss + z_loss

        return MoELossOutput(
            total_loss=total_loss,
            lm_loss=lm_loss,
            balance_loss=balance_loss,
            z_loss=z_loss,
        )

    def compute_balance_loss_only(
        self,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only the balance loss (for analysis)."""
        return self.balance_loss_fn(routing_weights)

    def get_loss_dict(
        self,
        lm_loss: torch.Tensor,
        routing_weights: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get loss as dict for logging.

        Returns dict with 'total_loss', 'lm_loss', 'balance_loss', 'z_loss'.
        """
        output = self.forward(lm_loss, routing_weights, router_logits)

        result = {
            'total_loss': output.total_loss,
            'lm_loss': output.lm_loss,
            'balance_loss': output.balance_loss,
        }
        if output.z_loss is not None:
            result['z_loss'] = output.z_loss

        return result


def compute_expert_utilization(
    routing_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute expert utilization (fraction of tokens per expert).

    Args:
        routing_weights: [B, num_experts] Routing probabilities
        num_experts: Number of experts

    Returns:
        utilization: [num_experts] Fraction of samples routed to each expert
    """
    dominant_experts = routing_weights.argmax(dim=-1)  # [B]

    utilization = torch.zeros(num_experts, device=routing_weights.device)
    for i in range(num_experts):
        utilization[i] = (dominant_experts == i).float().mean()

    return utilization


def compute_load_balance_coefficient(utilization: torch.Tensor) -> float:
    """
    Compute load balance coefficient (0 = all to one expert, 1 = uniform).

    Args:
        utilization: [num_experts] Expert utilization

    Returns:
        coefficient: Float in [0, 1]
    """
    num_experts = utilization.size(0)

    # Perfectly balanced would have utilization = 1/num_experts for all
    ideal = 1.0 / num_experts

    # Compute variance from ideal
    variance = ((utilization - ideal) ** 2).mean()

    # Max variance (all to one expert) = (1 - 1/n)^2 * 1/n + (0 - 1/n)^2 * (n-1)/n
    # Simplified: max_var = (n-1) / n^2
    max_variance = (num_experts - 1) / (num_experts ** 2)

    # Coefficient = 1 - normalized_variance
    if max_variance > 0:
        coefficient = 1.0 - (variance / max_variance).item()
    else:
        coefficient = 1.0

    return max(0.0, min(1.0, coefficient))


if __name__ == "__main__":
    print("Testing MoE Loss modules...")
    print()

    num_experts = 4
    batch_size = 16

    # Test LoadBalanceLoss
    print("Test 1: LoadBalanceLoss (MSE)...")
    balance_loss = LoadBalanceLoss(num_experts=num_experts, balance_type='mse')

    # Uniform routing should have low loss
    uniform_routing = torch.ones(batch_size, num_experts) / num_experts
    loss_uniform = balance_loss(uniform_routing)
    print(f"  Uniform routing loss: {loss_uniform.item():.6f}")

    # Skewed routing should have high loss
    skewed_routing = torch.zeros(batch_size, num_experts)
    skewed_routing[:, 0] = 1.0  # All to expert 0
    loss_skewed = balance_loss(skewed_routing)
    print(f"  Skewed routing loss: {loss_skewed.item():.6f}")

    assert loss_uniform < loss_skewed, "Uniform should have lower loss than skewed"
    print("  PASSED")

    # Test Switch balance loss
    print("\nTest 2: LoadBalanceLoss (Switch)...")
    switch_loss = LoadBalanceLoss(num_experts=num_experts, balance_type='switch')

    loss_uniform_switch = switch_loss(uniform_routing)
    loss_skewed_switch = switch_loss(skewed_routing)
    print(f"  Uniform routing loss: {loss_uniform_switch.item():.6f}")
    print(f"  Skewed routing loss: {loss_skewed_switch.item():.6f}")
    print("  PASSED")

    # Test Entropy balance loss
    print("\nTest 3: LoadBalanceLoss (Entropy)...")
    entropy_loss = LoadBalanceLoss(num_experts=num_experts, balance_type='entropy')

    loss_uniform_ent = entropy_loss(uniform_routing)
    loss_skewed_ent = entropy_loss(skewed_routing)
    print(f"  Uniform routing loss: {loss_uniform_ent.item():.6f}")
    print(f"  Skewed routing loss: {loss_skewed_ent.item():.6f}")

    assert loss_uniform_ent < loss_skewed_ent, "Uniform should have lower loss than skewed"
    print("  PASSED")

    # Test MoELoss
    print("\nTest 4: MoELoss...")
    moe_loss = MoELoss(
        num_experts=num_experts,
        balance_weight=0.1,
        balance_type='mse',
        z_loss_weight=0.001,
    )

    lm_loss = torch.tensor(2.5)  # Simulated LM loss
    routing_weights = F.softmax(torch.randn(batch_size, num_experts), dim=-1)
    router_logits = torch.randn(batch_size, num_experts)

    output = moe_loss(lm_loss, routing_weights, router_logits)
    print(f"  Total loss: {output.total_loss.item():.4f}")
    print(f"  LM loss: {output.lm_loss.item():.4f}")
    print(f"  Balance loss: {output.balance_loss.item():.6f}")
    if output.z_loss is not None:
        print(f"  Z-loss: {output.z_loss.item():.6f}")
    print("  PASSED")

    # Test utilization
    print("\nTest 5: Expert utilization...")
    utilization = compute_expert_utilization(routing_weights, num_experts)
    print(f"  Utilization: {utilization.tolist()}")
    print(f"  Sum: {utilization.sum().item():.4f}")

    coefficient = compute_load_balance_coefficient(utilization)
    print(f"  Balance coefficient: {coefficient:.4f}")
    print("  PASSED")

    # Test gradient flow
    print("\nTest 6: Gradient flow...")
    routing_weights = F.softmax(torch.randn(batch_size, num_experts), dim=-1)
    routing_weights.requires_grad = True
    router_logits = torch.randn(batch_size, num_experts, requires_grad=True)
    lm_loss = torch.tensor(2.5, requires_grad=True)

    output = moe_loss(lm_loss, routing_weights, router_logits)
    output.total_loss.backward()

    assert routing_weights.grad is not None, "No gradient for routing_weights"
    assert router_logits.grad is not None, "No gradient for router_logits"
    print("  PASSED")

    print()
    print("=== All tests passed! ===")
