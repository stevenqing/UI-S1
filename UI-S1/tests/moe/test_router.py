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
Tests for MoE Router module.

Run with: pytest tests/moe/test_router.py -v
"""

import pytest
import torch
import torch.nn as nn

from verl.models.moe.router import (
    TextOnlyRouter,
    RouterOutput,
    InstructionFeatureExtractor,
    compute_routing_entropy,
    compute_routing_diversity,
)


class TestRouterOutput:
    """Tests for RouterOutput dataclass."""

    def test_router_output_creation(self):
        """Test RouterOutput can be created with tensors."""
        batch_size = 4
        num_experts = 4
        top_k = 1

        output = RouterOutput(
            routing_weights=torch.randn(batch_size, num_experts),
            top_k_weights=torch.randn(batch_size, top_k),
            top_k_indices=torch.randint(0, num_experts, (batch_size, top_k)),
            router_logits=torch.randn(batch_size, num_experts),
        )

        assert output.routing_weights.shape == (batch_size, num_experts)
        assert output.top_k_weights.shape == (batch_size, top_k)
        assert output.top_k_indices.shape == (batch_size, top_k)

    def test_router_output_to_device(self):
        """Test RouterOutput can move to device."""
        output = RouterOutput(
            routing_weights=torch.randn(4, 4),
            top_k_weights=torch.randn(4, 1),
            top_k_indices=torch.randint(0, 4, (4, 1)),
            router_logits=torch.randn(4, 4),
        )

        # Move to CPU (should work even if already on CPU)
        output_cpu = output.to('cpu')
        assert output_cpu.routing_weights.device.type == 'cpu'

    def test_router_output_detach(self):
        """Test RouterOutput detach method."""
        output = RouterOutput(
            routing_weights=torch.randn(4, 4, requires_grad=True),
            top_k_weights=torch.randn(4, 1, requires_grad=True),
            top_k_indices=torch.randint(0, 4, (4, 1)),
            router_logits=torch.randn(4, 4, requires_grad=True),
        )

        detached = output.detach()
        assert not detached.routing_weights.requires_grad
        assert not detached.top_k_weights.requires_grad


class TestTextOnlyRouter:
    """Tests for TextOnlyRouter module."""

    @pytest.fixture
    def router(self):
        """Create a router for testing."""
        return TextOnlyRouter(
            hidden_size=256,
            num_experts=4,
            router_hidden=64,
            top_k=1,
        )

    @pytest.fixture
    def router_topk2(self):
        """Create a router with top_k=2."""
        return TextOnlyRouter(
            hidden_size=256,
            num_experts=4,
            router_hidden=64,
            top_k=2,
        )

    def test_router_output_shapes(self, router):
        """Test that router outputs have correct shapes."""
        batch_size = 8
        features = torch.randn(batch_size, 256)

        output = router(features)

        assert output.routing_weights.shape == (batch_size, 4)
        assert output.top_k_weights.shape == (batch_size, 1)
        assert output.top_k_indices.shape == (batch_size, 1)
        assert output.router_logits.shape == (batch_size, 4)

    def test_router_topk2_shapes(self, router_topk2):
        """Test router with top_k=2."""
        batch_size = 8
        features = torch.randn(batch_size, 256)

        output = router_topk2(features)

        assert output.top_k_weights.shape == (batch_size, 2)
        assert output.top_k_indices.shape == (batch_size, 2)

    def test_routing_weights_sum_to_one(self, router):
        """Test that routing weights sum to 1."""
        features = torch.randn(8, 256)
        output = router(features)

        sums = output.routing_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_topk_weights_sum_to_one(self, router_topk2):
        """Test that top-k weights sum to 1 after renormalization."""
        features = torch.randn(8, 256)
        output = router_topk2(features)

        sums = output.top_k_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_routing_weights_are_probabilities(self, router):
        """Test that routing weights are valid probabilities."""
        features = torch.randn(8, 256)
        output = router(features)

        assert (output.routing_weights >= 0).all()
        assert (output.routing_weights <= 1).all()

    def test_topk_indices_valid(self, router):
        """Test that top-k indices are valid expert indices."""
        features = torch.randn(8, 256)
        output = router(features)

        assert (output.top_k_indices >= 0).all()
        assert (output.top_k_indices < 4).all()

    def test_router_is_differentiable(self, router):
        """Test that router supports backpropagation."""
        features = torch.randn(8, 256, requires_grad=True)
        output = router(features)

        # Should be able to compute gradients
        loss = output.routing_weights.mean()
        loss.backward()

        assert features.grad is not None
        assert features.grad.shape == features.shape

    def test_router_deterministic(self, router):
        """Test that router is deterministic in eval mode."""
        router.eval()
        features = torch.randn(8, 256)

        output1 = router(features)
        output2 = router(features)

        assert torch.equal(output1.routing_weights, output2.routing_weights)
        assert torch.equal(output1.top_k_indices, output2.top_k_indices)

    def test_get_routing_distribution(self, router):
        """Test get_routing_distribution method."""
        features = torch.randn(8, 256)
        distribution = router.get_routing_distribution(features)

        assert distribution.shape == (8, 4)
        assert torch.allclose(distribution.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_get_hard_routing(self, router):
        """Test get_hard_routing method."""
        features = torch.randn(8, 256)
        hard_routing = router.get_hard_routing(features)

        assert hard_routing.shape == (8,)
        assert (hard_routing >= 0).all()
        assert (hard_routing < 4).all()

    def test_temperature_effect(self):
        """Test that temperature affects routing sharpness."""
        features = torch.randn(8, 256)

        router_low_temp = TextOnlyRouter(hidden_size=256, num_experts=4, temperature=0.1)
        router_high_temp = TextOnlyRouter(hidden_size=256, num_experts=4, temperature=2.0)

        # Copy weights so they produce same logits
        router_high_temp.load_state_dict(router_low_temp.state_dict())

        output_low = router_low_temp(features)
        output_high = router_high_temp(features)

        # Low temperature should have higher max probability (sharper)
        max_low = output_low.routing_weights.max(dim=-1).values.mean()
        max_high = output_high.routing_weights.max(dim=-1).values.mean()

        assert max_low > max_high

    def test_invalid_topk(self):
        """Test that invalid top_k raises error."""
        with pytest.raises(ValueError):
            TextOnlyRouter(hidden_size=256, num_experts=4, top_k=5)


class TestInstructionFeatureExtractor:
    """Tests for InstructionFeatureExtractor module."""

    @pytest.fixture
    def hidden_states(self):
        """Create sample hidden states."""
        return torch.randn(4, 512, 256)

    @pytest.fixture
    def instruction_mask(self):
        """Create sample instruction mask."""
        mask = torch.zeros(4, 512, dtype=torch.bool)
        mask[:, 100:150] = True  # Instruction at positions 100-150
        return mask

    def test_mean_pooling(self, hidden_states, instruction_mask):
        """Test mean pooling strategy."""
        extractor = InstructionFeatureExtractor(pooling_strategy='mean')
        features = extractor(hidden_states, instruction_mask)

        assert features.shape == (4, 256)

    def test_last_pooling(self, hidden_states, instruction_mask):
        """Test last token pooling strategy."""
        extractor = InstructionFeatureExtractor(pooling_strategy='last')
        features = extractor(hidden_states, instruction_mask)

        assert features.shape == (4, 256)

    def test_first_pooling(self, hidden_states, instruction_mask):
        """Test first token pooling strategy."""
        extractor = InstructionFeatureExtractor(pooling_strategy='first')
        features = extractor(hidden_states, instruction_mask)

        assert features.shape == (4, 256)

    def test_max_pooling(self, hidden_states, instruction_mask):
        """Test max pooling strategy."""
        extractor = InstructionFeatureExtractor(pooling_strategy='max')
        features = extractor(hidden_states, instruction_mask)

        assert features.shape == (4, 256)

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            InstructionFeatureExtractor(pooling_strategy='invalid')

    def test_empty_mask_fallback(self, hidden_states):
        """Test fallback when mask is all False."""
        extractor = InstructionFeatureExtractor(pooling_strategy='mean')
        empty_mask = torch.zeros(4, 512, dtype=torch.bool)

        # Should not raise error, use fallback
        features = extractor(hidden_states, empty_mask)
        assert features.shape == (4, 256)

    def test_with_projection(self, hidden_states, instruction_mask):
        """Test extractor with projection layer."""
        extractor = InstructionFeatureExtractor(
            pooling_strategy='mean',
            hidden_size=256,
            output_size=128,
        )
        features = extractor(hidden_states, instruction_mask)

        assert features.shape == (4, 128)


class TestRoutingUtilities:
    """Tests for routing utility functions."""

    def test_compute_routing_entropy(self):
        """Test routing entropy computation."""
        # Uniform distribution should have max entropy
        uniform = torch.ones(8, 4) / 4
        entropy_uniform = compute_routing_entropy(uniform)

        # Sharp distribution should have low entropy
        sharp = torch.zeros(8, 4)
        sharp[:, 0] = 1.0
        entropy_sharp = compute_routing_entropy(sharp)

        assert (entropy_uniform > entropy_sharp).all()

    def test_compute_routing_diversity(self):
        """Test routing diversity computation."""
        # All samples to same expert = low diversity
        same_expert = torch.zeros(8, 4)
        same_expert[:, 0] = 1.0
        diversity_same = compute_routing_diversity(same_expert)

        # Different experts = high diversity
        different = torch.zeros(8, 4)
        for i in range(8):
            different[i, i % 4] = 1.0
        diversity_different = compute_routing_diversity(different)

        assert diversity_different > diversity_same


class TestRouterIntegration:
    """Integration tests for router with realistic scenarios."""

    def test_router_with_qwen_hidden_size(self):
        """Test router with Qwen2.5-VL hidden size."""
        router = TextOnlyRouter(
            hidden_size=3584,  # Qwen2.5-VL-7B
            num_experts=4,
            router_hidden=256,
        )

        features = torch.randn(8, 3584)
        output = router(features)

        assert output.routing_weights.shape == (8, 4)

    def test_router_parameter_count(self):
        """Test that router has expected parameter count."""
        hidden_size = 256
        router_hidden = 64
        num_experts = 4

        router = TextOnlyRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_hidden=router_hidden,
        )

        # Expected: Linear(256->64) + Linear(64->4)
        # = 256*64 + 64 + 64*4 + 4 = 16,384 + 64 + 256 + 4 = 16,708
        expected_params = (hidden_size * router_hidden + router_hidden +
                          router_hidden * num_experts + num_experts)

        actual_params = sum(p.numel() for p in router.parameters())

        assert actual_params == expected_params

    def test_router_gradient_flow(self):
        """Test that gradients flow through the router correctly."""
        router = TextOnlyRouter(hidden_size=256, num_experts=4)
        features = torch.randn(8, 256, requires_grad=True)

        output = router(features)

        # Create a loss that depends on routing
        loss = (output.routing_weights * torch.randn(8, 4)).sum()
        loss.backward()

        # Check gradients exist for router parameters
        for name, param in router.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
