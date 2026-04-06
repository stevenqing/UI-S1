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
FPseudo Reward Manager: extends DAPORewardManager with f_pseudo reward shaping.

f_pseudo(t) = f(s_t) - f(s_{t+1}) provides an additional reward signal that
encourages the agent to cross connectivity bottlenecks (from high-f to low-f regions).

Total reward: r_total = r_action_match + lambda * f_pseudo

Usage:
    In config YAML:
        reward_model:
            reward_manager: f_pseudo_dapo
            reward_kwargs:
                f_pseudo_path: outputs/f_pseudo/f_pseudo_map.json
                f_pseudo_lambda: 0.1
"""

import json
import logging
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.dapo import DAPORewardManager

logger = logging.getLogger(__name__)


@register("f_pseudo_dapo")
class FPseudoDAPORewardManager(DAPORewardManager):
    """Reward manager that adds f_pseudo reward shaping to the base DAPO reward.

    The f_pseudo signal comes from precomputed eigenfunction values:
      f_pseudo(t) = f(s_t) - f(s_{t+1})

    When the agent crosses a connectivity bottleneck (moving from high-f to low-f),
    f_pseudo > 0, providing a bonus reward.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        f_pseudo_path: str = "",
        f_pseudo_lambda: float = 0.1,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=overlong_buffer_cfg,
        )
        self.f_pseudo_lambda = f_pseudo_lambda
        self.f_pseudo_map: dict[str, dict[str, float]] = {}
        self._f_pseudo_stats = {"lookups": 0, "hits": 0, "misses": 0}

        if f_pseudo_path:
            self._load_f_pseudo_map(f_pseudo_path)

    def _load_f_pseudo_map(self, path: str) -> None:
        """Load precomputed f_pseudo values from JSON file."""
        try:
            with open(path) as f:
                self.f_pseudo_map = json.load(f)
            total_steps = sum(len(v) for v in self.f_pseudo_map.values())
            logger.info(
                f"Loaded f_pseudo_map from {path}: "
                f"{len(self.f_pseudo_map)} trajectories, {total_steps} steps"
            )
        except FileNotFoundError:
            logger.warning(f"f_pseudo_map not found at {path}, f_pseudo will be 0 for all steps")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse f_pseudo_map at {path}: {e}")

    def _lookup_f_pseudo(self, execution_id: str, step_id: int | str) -> float:
        """Look up f_pseudo value for a given (execution_id, step_id) pair."""
        self._f_pseudo_stats["lookups"] += 1
        step_key = str(step_id)
        traj_map = self.f_pseudo_map.get(execution_id)
        if traj_map is not None:
            val = traj_map.get(step_key)
            if val is not None:
                self._f_pseudo_stats["hits"] += 1
                return float(val)
        self._f_pseudo_stats["misses"] += 1
        return 0.0

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute reward with f_pseudo shaping added to the base score."""

        # If rm_scores already computed, pass through
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # --- f_pseudo reward shaping ---
            # f_pseudo_map uses 1-indexed step keys ("1","2",...) from transitions.jsonl,
            # while the RL trainer sets step_id = si (0-indexed).
            # Fix: map si → si+1 for lookup, and skip the last step (no next state).
            f_pseudo = 0.0
            if self.f_pseudo_map and self.f_pseudo_lambda != 0.0:
                execution_id = data_item.non_tensor_batch.get("execution_id", "")
                step_id = data_item.non_tensor_batch.get("step_id", -1)
                num_steps = ground_truth.get("num_steps", 0) if isinstance(ground_truth, dict) else 0
                if execution_id and step_id >= 0:
                    # Last step has no next state → no f_pseudo
                    if num_steps > 0 and step_id >= num_steps - 1:
                        f_pseudo = 0.0
                    else:
                        f_pseudo = self._lookup_f_pseudo(execution_id, step_id + 1)

            f_pseudo_bonus = self.f_pseudo_lambda * f_pseudo
            reward += f_pseudo_bonus

            reward_extra_info["f_pseudo"].append(f_pseudo)
            reward_extra_info["f_pseudo_bonus"].append(f_pseudo_bonus)
            reward_extra_info["base_reward"].append(score)
            # --- end f_pseudo ---

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                print(f"[f_pseudo]", f_pseudo)
                print(f"[f_pseudo_bonus]", f_pseudo_bonus)
                print(f"[total_reward]", reward)

        # Log f_pseudo statistics periodically
        if self._f_pseudo_stats["lookups"] > 0 and self._f_pseudo_stats["lookups"] % 1000 < len(data):
            hit_rate = self._f_pseudo_stats["hits"] / max(self._f_pseudo_stats["lookups"], 1) * 100
            logger.info(
                f"f_pseudo lookup stats: {self._f_pseudo_stats['hits']}/{self._f_pseudo_stats['lookups']} "
                f"hits ({hit_rate:.1f}%)"
            )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
