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
MoE (Mixture of Experts) modules for GUI Agent.

This package contains:
- Router: Routes instructions to appropriate experts
- Expert LoRA: Collection of expert LoRA adapters
- MoE Loss: Load balance and auxiliary losses
- MoE Wrapper: Integrates router and experts with base VLM
- State Representation: State hashing and embedding for graph-based option discovery

Usage:
    from verl.models.moe import MoEVLMWrapper, MoEConfig

    # Create MoE wrapper around base model
    config = MoEConfig(num_experts=4, expert_lora_r=16)
    moe_model = MoEVLMWrapper(base_model, config)

    # Forward pass
    output = moe_model(input_ids, attention_mask, pixel_values, labels=labels)
    output.loss.backward()
"""

from verl.models.moe.router import (
    TextOnlyRouter,
    RouterOutput,
    InstructionFeatureExtractor,
    create_instruction_mask,
    create_instruction_mask_from_text,
    compute_routing_entropy,
    compute_routing_diversity,
)

from verl.models.moe.expert_lora import (
    LoRALayer,
    SingleExpertLoRA,
    ExpertLoRACollection,
    ExpertLoRAConfig,
    MoEExpertApplier,
    compute_expert_parameter_count,
)

from verl.models.moe.moe_loss import (
    MoELoss,
    MoELossOutput,
    LoadBalanceLoss,
    RouterZLoss,
    compute_expert_utilization,
    compute_load_balance_coefficient,
)

from verl.models.moe.moe_wrapper import (
    MoEVLMWrapper,
    MoEConfig,
    MoEOutput,
    create_moe_wrapper,
)

from verl.models.moe.graph_analysis import (
    EigenfunctionNet,
    EigenfunctionConfig,
    TransitionPairDataset,
    eigenfunction_loss,
    train_eigenfunction,
    identify_bottlenecks,
    map_bottlenecks_to_descriptions,
    load_f_net,
)

from verl.models.moe.vlm_eigenfunction import (
    VLMEigenfunctionConfig,
    VLMEigenfunctionModel,
    RegressionHead,
    VLMTransitionDataset,
    VLMStateDataset,
    VLMCollator,
    vlm_eigenfunction_loss,
)

from verl.models.moe.state_representation import (
    GUI360StepData,
    StateID,
    TransitionRecord,
    GUI360TrajectoryProcessor,
    extract_state_id,
    extract_state_id_from_raw,
    extract_state_embedding,
    extract_state_embedding_from_raw,
    get_embedding_dim,
    save_transitions,
    load_transitions,
)

__all__ = [
    # Router
    "TextOnlyRouter",
    "RouterOutput",
    "InstructionFeatureExtractor",
    "create_instruction_mask",
    "create_instruction_mask_from_text",
    "compute_routing_entropy",
    "compute_routing_diversity",
    # Expert LoRA
    "LoRALayer",
    "SingleExpertLoRA",
    "ExpertLoRACollection",
    "ExpertLoRAConfig",
    "MoEExpertApplier",
    "compute_expert_parameter_count",
    # MoE Loss
    "MoELoss",
    "MoELossOutput",
    "LoadBalanceLoss",
    "RouterZLoss",
    "compute_expert_utilization",
    "compute_load_balance_coefficient",
    # MoE Wrapper
    "MoEVLMWrapper",
    "MoEConfig",
    "MoEOutput",
    "create_moe_wrapper",
    # State Representation
    "GUI360StepData",
    "StateID",
    "TransitionRecord",
    "GUI360TrajectoryProcessor",
    "extract_state_id",
    "extract_state_id_from_raw",
    "extract_state_embedding",
    "extract_state_embedding_from_raw",
    "get_embedding_dim",
    "save_transitions",
    "load_transitions",
    # Graph Analysis (Eigenfunction)
    "EigenfunctionNet",
    "EigenfunctionConfig",
    "TransitionPairDataset",
    "eigenfunction_loss",
    "train_eigenfunction",
    "identify_bottlenecks",
    "map_bottlenecks_to_descriptions",
    "load_f_net",
    # VLM Eigenfunction (Task 3.1)
    "VLMEigenfunctionConfig",
    "VLMEigenfunctionModel",
    "RegressionHead",
    "VLMTransitionDataset",
    "VLMStateDataset",
    "VLMCollator",
    "vlm_eigenfunction_loss",
]
