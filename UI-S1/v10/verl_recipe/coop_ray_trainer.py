"""Cooperative LoRA Ray Trainer.

Extends RayPPOTrainer with a dual-phase fit() loop:
  Phase 1: Grounder rollout → grounder descriptions
  Phase 2: Actor rollout → actions (conditioned on grounder output)
  Phase 3: Dual reward + dual advantage computation
  Phase 4: Dual update (grounder update, then actor update)

The driver orchestrates all phases; workers handle the distributed compute.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.debug.performance import _timer
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean

from v10.verl_recipe.coop_dataset import (
    build_actor_messages,
    build_actor_messages_structured,
    format_actor_text,
    ACTOR_SYSTEM,
)
from v10.verl_recipe.reward_fn import grounder_reward, actor_reward


class CoopRayTrainer(RayPPOTrainer):
    """Dual-adapter cooperative GRPO trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coop_config = self.config.actor_rollout_ref.get("cooperative", {})

    def fit(self):
        """The dual-phase cooperative GRPO training loop."""
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        # Validate before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
            # Free fragmented CUDA memory on workers after validation
            self.actor_rollout_wg.free_cache()

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Coop Training",
        )

        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # ════════════════════════════════════════════════
                    # Phase 1: Grounder Rollout
                    # ════════════════════════════════════════════════
                    with _timer("grounder_gen", timing_raw):
                        g_batch, g_gen_output = self._grounder_rollout(batch)

                    # ════════════════════════════════════════════════
                    # Phase 2: Actor Rollout (conditioned on grounder)
                    # ════════════════════════════════════════════════
                    with _timer("actor_gen", timing_raw):
                        a_batch, a_gen_output = self._actor_rollout(batch, g_batch)

                    # ════════════════════════════════════════════════
                    # Phase 3: Reward + Advantage
                    # ════════════════════════════════════════════════
                    with _timer("reward", timing_raw):
                        g_batch, a_batch, reward_metrics = self._compute_dual_rewards(
                            g_batch, a_batch, batch
                        )
                    metrics.update(reward_metrics)

                    with _timer("advantage", timing_raw):
                        g_batch = self._compute_grpo_advantage(g_batch, "grounder")
                        a_batch = self._compute_grpo_advantage(a_batch, "actor")

                    # ════════════════════════════════════════════════
                    # Phase 4: Log Probs (old + ref)
                    # ════════════════════════════════════════════════
                    with _timer("log_probs", timing_raw):
                        g_batch = self._compute_old_and_ref_log_probs(
                            g_batch, adapter_name="default"
                        )
                        a_batch = self._compute_old_and_ref_log_probs(
                            a_batch, adapter_name="actor_lora"
                        )

                    # ════════════════════════════════════════════════
                    # Phase 5: Dual Policy Update
                    # ════════════════════════════════════════════════
                    with _timer("update", timing_raw):
                        # Grounder update
                        g_batch.meta_info["adapter_name"] = "default"
                        g_batch.meta_info["multi_turn"] = False
                        g_batch.meta_info["global_token_num"] = torch.sum(
                            g_batch.batch["attention_mask"], dim=-1
                        ).tolist()
                        g_output = self.actor_rollout_wg.update_actor_with_adapter(g_batch)
                        g_update_metrics = reduce_metrics(g_output.meta_info.get("metrics", {}))
                        metrics.update({f"grounder/{k}": v for k, v in g_update_metrics.items()})

                        # Actor update
                        a_batch.meta_info["adapter_name"] = "actor_lora"
                        a_batch.meta_info["multi_turn"] = False
                        a_batch.meta_info["global_token_num"] = torch.sum(
                            a_batch.batch["attention_mask"], dim=-1
                        ).tolist()
                        a_output = self.actor_rollout_wg.update_actor_with_adapter(a_batch)
                        a_update_metrics = reduce_metrics(a_output.meta_info.get("metrics", {}))
                        metrics.update({f"actor/{k}": v for k, v in a_update_metrics.items()})

                    # ════════════════════════════════════════════════
                    # Validation & Checkpoint
                    # ════════════════════════════════════════════════
                    if (self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                        metrics.update(val_metrics)
                        self.actor_rollout_wg.free_cache()

                    if (self.config.trainer.save_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0)):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # Timing metrics
                for name, val in timing_raw.items():
                    metrics[f"timing_s/{name}"] = val
                metrics["train/global_step"] = self.global_steps

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "g_r": metrics.get("reward/grounder_mean", 0),
                    "a_r": metrics.get("reward/actor_mean", 0),
                })

                self.global_steps += 1
                if self.global_steps > self.total_training_steps:
                    break

            if self.global_steps > self.total_training_steps:
                break

        progress_bar.close()

    # ── Phase 1: Grounder Rollout ───────────────────────────────────────

    def _grounder_rollout(self, batch: DataProto):
        """Generate K grounder descriptions per sample."""
        n = self.config.actor_rollout_ref.rollout.n  # K

        # Repeat each sample K times
        g_batch = batch.repeat(repeat_times=n, interleave=True)

        # Pop keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in g_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in g_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")

        gen_batch = g_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch.meta_info["n"] = 1
        gen_batch.meta_info["adapter_name"] = "default"  # grounder adapter

        # Generate
        gen_output = self.actor_rollout_wg.generate_with_adapter(gen_batch)
        timing = gen_output.meta_info.pop("timing", {})

        # Assign UIDs for GRPO grouping
        # Each group of K samples from the same prompt shares a UID
        batch_size = len(batch.batch["input_ids"]) if "input_ids" in batch.batch.keys() else len(batch)
        g_batch.non_tensor_batch["uid"] = self._make_group_uids(batch_size, n)

        g_batch = g_batch.union(gen_output)
        g_batch.batch["response_mask"] = compute_response_mask(g_batch)

        return g_batch, gen_output

    def _make_group_uids(self, batch_size: int, n: int) -> np.ndarray:
        """Create UIDs where every N consecutive samples share the same UID."""
        uids = []
        for i in range(batch_size):
            uid = str(uuid.uuid4())
            uids.extend([uid] * n)
        return np.array(uids, dtype=object)

    # ── Phase 2: Actor Rollout ──────────────────────────────────────────

    def _actor_rollout(self, original_batch: DataProto, g_batch: DataProto):
        """Generate actor actions conditioned on grounder outputs."""
        n = self.config.actor_rollout_ref.rollout.n

        # Decode grounder responses
        grounder_texts = self._decode_responses(g_batch)

        # Build actor prompts
        a_batch = self._build_actor_batch(original_batch, g_batch, grounder_texts, n)

        # Pop keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in a_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in a_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")

        gen_batch = a_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch.meta_info["n"] = 1
        gen_batch.meta_info["adapter_name"] = "actor_lora"

        # Generate
        gen_output = self.actor_rollout_wg.generate_with_adapter(gen_batch)
        gen_output.meta_info.pop("timing", {})

        # Copy UIDs from grounder batch (same grouping)
        a_batch.non_tensor_batch["uid"] = g_batch.non_tensor_batch["uid"].copy()

        a_batch = a_batch.union(gen_output)
        a_batch.batch["response_mask"] = compute_response_mask(a_batch)

        return a_batch, gen_output

    def _decode_responses(self, batch: DataProto) -> list:
        """Decode response token IDs to text strings."""
        responses = batch.batch["responses"]
        texts = []
        for i in range(len(responses)):
            resp_ids = responses[i]
            # Find valid (non-pad) tokens
            attn = batch.batch["attention_mask"][i]
            prompt_len = batch.batch["prompts"][i].shape[0] if "prompts" in batch.batch.keys() else 0
            # Decode
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            texts.append(text)
        return texts

    def _build_actor_batch(
        self,
        original_batch: DataProto,
        g_batch: DataProto,
        grounder_texts: list,
        n: int,
    ) -> DataProto:
        """Build actor DataProto with grounder outputs as context.

        For each of the batch_size * n samples, construct an actor prompt
        that includes the corresponding grounder output.
        """
        total_samples = len(grounder_texts)

        # Extract per-sample metadata from original batch
        # original_batch has batch_size samples, g_batch has batch_size * n
        all_input_ids = []
        all_attention_masks = []
        all_position_ids = []
        all_raw_prompts = []
        all_multi_modal_data = []
        all_non_tensors = defaultdict(list)

        for i in range(total_samples):
            orig_idx = i // n  # Map back to original sample index

            # Get sample info from original batch's extra_info
            extra_info = original_batch.non_tensor_batch.get("extra_info", [{}])[orig_idx]
            if isinstance(extra_info, str):
                extra_info = json.loads(extra_info)

            goal = extra_info.get("goal", "")
            history = extra_info.get("history", "")
            image_path = extra_info.get("image_path", "")
            grounding = grounder_texts[i]

            # Build actor messages (structured format for proper image tokenization)
            messages = build_actor_messages_structured(image_path, goal, history, grounding)

            # Tokenize using processor
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # Process with images
            from PIL import Image as PILImage
            try:
                image = PILImage.open(image_path).convert("RGB")
                model_inputs = self.processor(
                    text=[raw_prompt], images=[image], return_tensors="pt"
                )
            except Exception:
                # Fallback: text-only
                model_inputs = self.processor(
                    text=[raw_prompt], return_tensors="pt"
                )

            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]

            # Pad/truncate to max_prompt_length
            from verl.utils.torch_functional import postprocess_data
            max_len = self.config.data.get("max_prompt_length", 4096)
            input_ids, attention_mask = postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_len,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="left",
            )

            # Position IDs
            from verl.utils.model import compute_position_id_with_mask
            if hasattr(self.processor, 'image_processor') and \
               "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                from verl.models.transformers.qwen2_vl import get_rope_index
                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    attention_mask=attention_mask[0],
                )
            else:
                position_ids = compute_position_id_with_mask(attention_mask)[0]

            all_input_ids.append(input_ids[0])
            all_attention_masks.append(attention_mask[0])
            all_position_ids.append(position_ids)
            all_raw_prompts.append(raw_prompt)

            # Multi-modal data
            mm_data = {}
            try:
                image = PILImage.open(image_path).convert("RGB")
                mm_data["image"] = [image]
            except Exception:
                pass
            all_multi_modal_data.append(mm_data)

            # Multi-modal inputs (pixel_values, image_grid_thw etc.)
            mm_inputs = {}
            for k in model_inputs:
                if k not in ("input_ids", "attention_mask"):
                    mm_inputs[k] = model_inputs[k]

            # Copy non-tensor fields from original batch
            for key in original_batch.non_tensor_batch:
                if key in ("uid",):
                    continue
                val = original_batch.non_tensor_batch[key][orig_idx]
                all_non_tensors[key].append(val)

        # Stack tensors — need to pad to same length
        max_seq_len = max(ids.shape[-1] for ids in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_position_ids = []

        for ids, mask, pos in zip(all_input_ids, all_attention_masks, all_position_ids):
            seq_len = ids.shape[-1]
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                ids = torch.cat([
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=ids.dtype),
                    ids
                ])
                mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
                if pos.dim() == 2:  # (3, seq_len) for Qwen2VL
                    pos = torch.cat([torch.zeros(pos.shape[0], pad_len, dtype=pos.dtype), pos], dim=-1)
                else:
                    pos = torch.cat([torch.zeros(pad_len, dtype=pos.dtype), pos])
            padded_input_ids.append(ids)
            padded_attention_masks.append(mask)
            padded_position_ids.append(pos)

        from tensordict import TensorDict as TD

        batch_tensors = TD({
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "position_ids": torch.stack(padded_position_ids),
        }, batch_size=[total_samples])

        non_tensor_batch = {}
        for key, vals in all_non_tensors.items():
            non_tensor_batch[key] = np.array(vals, dtype=object)
        non_tensor_batch["raw_prompt"] = np.array(all_raw_prompts, dtype=object)
        non_tensor_batch["multi_modal_data"] = np.array(all_multi_modal_data, dtype=object)

        raw_prompt_ids = [
            self.tokenizer.encode(rp, add_special_tokens=False)
            for rp in all_raw_prompts
        ]
        non_tensor_batch["raw_prompt_ids"] = np.array(raw_prompt_ids, dtype=object)

        return DataProto(batch=batch_tensors, non_tensor_batch=non_tensor_batch)

    # ── Phase 3: Dual Reward ────────────────────────────────────────────

    def _compute_dual_rewards(
        self,
        g_batch: DataProto,
        a_batch: DataProto,
        original_batch: DataProto,
    ):
        """Compute grounder and actor rewards from actor outputs."""
        n = self.config.actor_rollout_ref.rollout.n

        # Decode actor responses
        actor_texts = self._decode_responses(a_batch)

        # Get ground truth actions
        g_rewards_list = []
        a_rewards_list = []

        for i, actor_text in enumerate(actor_texts):
            orig_idx = i // n

            # Get GT action
            reward_model_info = original_batch.non_tensor_batch.get("reward_model", [{}])[orig_idx]
            if isinstance(reward_model_info, str):
                reward_model_info = json.loads(reward_model_info)
            gt_action = reward_model_info.get("ground_truth", {})
            if isinstance(gt_action, str):
                gt_action = json.loads(gt_action)

            # Get image dimensions
            extra_info = original_batch.non_tensor_batch.get("extra_info", [{}])[orig_idx]
            if isinstance(extra_info, str):
                extra_info = json.loads(extra_info)
            image_w = extra_info.get("image_w", 1080)
            image_h = extra_info.get("image_h", 2400)

            g_r = grounder_reward(actor_text, gt_action, image_w, image_h)
            a_r = actor_reward(actor_text, gt_action, image_w, image_h)

            g_rewards_list.append(g_r)
            a_rewards_list.append(a_r)

        # Build token-level reward tensors (reward at last valid token)
        g_response_len = g_batch.batch["responses"].shape[1]
        a_response_len = a_batch.batch["responses"].shape[1]

        g_reward_tensor = torch.zeros(
            len(g_rewards_list), g_response_len, dtype=torch.float32
        )
        a_reward_tensor = torch.zeros(
            len(a_rewards_list), a_response_len, dtype=torch.float32
        )

        g_response_mask = g_batch.batch["response_mask"]
        a_response_mask = a_batch.batch["response_mask"]

        for i in range(len(g_rewards_list)):
            # Place reward at last valid response token
            valid_len = int(g_response_mask[i].sum().item())
            if valid_len > 0:
                g_reward_tensor[i, valid_len - 1] = g_rewards_list[i]

        for i in range(len(a_rewards_list)):
            valid_len = int(a_response_mask[i].sum().item())
            if valid_len > 0:
                a_reward_tensor[i, valid_len - 1] = a_rewards_list[i]

        g_batch.batch["token_level_scores"] = g_reward_tensor
        g_batch.batch["token_level_rewards"] = g_reward_tensor
        a_batch.batch["token_level_scores"] = a_reward_tensor
        a_batch.batch["token_level_rewards"] = a_reward_tensor

        # Metrics
        metrics = {
            "reward/grounder_mean": float(np.mean(g_rewards_list)),
            "reward/actor_mean": float(np.mean(a_rewards_list)),
            "reward/grounder_std": float(np.std(g_rewards_list)),
            "reward/actor_std": float(np.std(a_rewards_list)),
            "reward/grounder_nonzero_frac": float(np.mean([r > 0 for r in g_rewards_list])),
            "reward/actor_nonzero_frac": float(np.mean([r > 0 for r in a_rewards_list])),
        }

        return g_batch, a_batch, metrics

    # ── Phase 3b: GRPO Advantage ────────────────────────────────────────

    def _compute_grpo_advantage(self, batch: DataProto, name: str) -> DataProto:
        """Compute GRPO advantages for one adapter's batch."""
        norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv,
        )

        # Log advantage stats
        adv = batch.batch["advantages"]
        response_mask = batch.batch["response_mask"]
        adv_abs = (adv * response_mask).abs().sum(-1) / response_mask.sum(-1).clamp(min=1)
        nonzero = (adv_abs > 1e-6).float().mean().item()

        return batch

    # ── Phase 4: Log Probs ──────────────────────────────────────────────

    def _compute_old_and_ref_log_probs(
        self, batch: DataProto, adapter_name: str
    ) -> DataProto:
        """Compute old log probs (current policy) and ref log probs (base model)."""
        # Old log probs with active adapter
        batch.meta_info["adapter_name"] = adapter_name
        old_log_prob = self.actor_rollout_wg.compute_log_prob_with_adapter(batch)
        if "entropys" in old_log_prob.batch.keys():
            old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)

        # Ref log probs (disable all adapters → base model)
        if self.config.actor_rollout_ref.actor.get("use_kl_loss", False):
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        return batch
