"""
OPD v2 (Optimal Patch Demonstration): Two-phase token-level signal for RL training.

Phase 1 — Safe Substitution (step_id=0 only):
  At step_id=0, all K rollouts share the exact same prompt (goal + GT screenshot).
  Replace incorrect rollouts' responses with best correct rollout's response.
  No advantage boost — just clean token-level gradient.

Phase 2 — Auxiliary Imitation Loss (all steps):
  For each (uid, step_id) group with mixed correct/incorrect rollouts,
  store the correct response as a target for an auxiliary cross-entropy loss.
  This provides token-level gradient independent of SPWA weighting,
  which is critical in failure zones where SPWA ≈ 0.

V7 — Visual Hindsight Conditioning (aux-loss-level s_{t+1} injection):
  For incorrect rollouts with s_{t+1} available, construct enriched prompts
  (original messages + s_{t+1} screenshot) and compute CE loss vs the best
  correct donor response. This is a Visual Inverse Dynamics Model signal:
  given (s_t, s_{t+1}), predict a_t. Rollouts stay 100% normal (no s_{t+1}),
  so inference distribution is preserved.

Must be called BEFORE old_log_prob computation so forward passes compute
correct log probs for the substituted tokens.
"""

import copy
import math
import traceback
from collections import defaultdict

import numpy as np
import torch


def compute_opd_substitution(batch, opd_fraction=0.5):
    """
    Phase 1: Response substitution at step_id=0 only.

    For each (uid, step_id=0) group with BOTH correct and incorrect rollouts:
    1. Pick best correct rollout as donor (highest sp_score)
    2. Select opd_fraction of incorrect rollouts as recipients (worst sp_score first)
    3. Replace recipients' responses + attention_mask with donor's
    4. Set extract_match=True for recipients

    Args:
        batch: DataProto with batch tensors and non_tensor_batch arrays.
        opd_fraction: Fraction of incorrect rollouts to substitute (0.0-1.0).

    Returns:
        batch: Modified in-place with substituted responses.
        metrics: Dict of OPD logging metrics.
    """
    extract_match = batch.non_tensor_batch['extract_match']
    uids = batch.non_tensor_batch['uid']
    step_ids = batch.non_tensor_batch['step_id']
    sp_scores = batch.non_tensor_batch.get('sp_scores', None)
    is_hindsight = batch.non_tensor_batch.get('is_hindsight', None)

    batch_size = len(extract_match)
    response_length = batch.batch['responses'].shape[1]

    # Group indices by (uid, step_id)
    groups = defaultdict(list)
    for i in range(batch_size):
        key = (uids[i], step_ids[i])
        groups[key].append(i)

    # Metrics
    n_eligible_groups = 0
    n_all_correct_groups = 0
    n_all_wrong_groups = 0
    n_substituted = 0
    n_skipped_nonzero_step = 0
    n_hindsight_skipped_recipient = 0

    for key, indices in groups.items():
        uid, step_id = key

        # Phase 1: only substitute at step_id == 0
        if step_id != 0:
            n_skipped_nonzero_step += 1
            continue

        if len(indices) <= 1:
            continue

        correct_idxs = [i for i in indices if extract_match[i]]
        incorrect_idxs = [i for i in indices if not extract_match[i]]

        if len(correct_idxs) == 0:
            n_all_wrong_groups += 1
            continue
        if len(incorrect_idxs) == 0:
            n_all_correct_groups += 1
            continue

        n_eligible_groups += 1

        # Pick best correct rollout as donor (highest sp_score)
        # Prefer hindsight correct rollouts as donors (higher quality signal)
        if sp_scores is not None:
            donor_idx = max(correct_idxs, key=lambda i: sp_scores[i])
        else:
            donor_idx = correct_idxs[0]

        # Skip hindsight rollouts as recipients — their prompts differ (extra image)
        # so token substitution would create a prompt-response mismatch
        if is_hindsight is not None:
            non_hindsight_incorrect = [i for i in incorrect_idxs if not is_hindsight[i]]
            n_hindsight_skipped_recipient += len(incorrect_idxs) - len(non_hindsight_incorrect)
            incorrect_idxs = non_hindsight_incorrect

        if len(incorrect_idxs) == 0:
            continue

        # Select worst incorrect rollouts as recipients
        n_recipients = max(1, math.ceil(len(incorrect_idxs) * opd_fraction))
        if sp_scores is not None:
            incorrect_idxs_sorted = sorted(incorrect_idxs, key=lambda i: sp_scores[i])
        else:
            incorrect_idxs_sorted = incorrect_idxs
        recipients = incorrect_idxs_sorted[:n_recipients]

        # Perform substitution
        for r_idx in recipients:
            batch.batch['responses'][r_idx] = batch.batch['responses'][donor_idx].clone()
            batch.batch['attention_mask'][r_idx, -response_length:] = \
                batch.batch['attention_mask'][donor_idx, -response_length:].clone()
            batch.non_tensor_batch['extract_match'][r_idx] = True
            n_substituted += 1

    metrics = {
        'opd/n_eligible_groups': n_eligible_groups,
        'opd/n_all_correct_groups': n_all_correct_groups,
        'opd/n_all_wrong_groups': n_all_wrong_groups,
        'opd/n_substituted_step0': n_substituted,
        'opd/n_skipped_nonzero_step': n_skipped_nonzero_step,
        'opd/n_hindsight_skipped_recipient': n_hindsight_skipped_recipient,
        'opd/substitution_rate': n_substituted / batch_size if batch_size > 0 else 0.0,
    }

    return batch, metrics


def compute_opd_targets(batch):
    """
    Phase 2: For each (uid, step_id) group, identify the correct response
    as an auxiliary imitation target for all incorrect rollouts.

    Works at ALL step_ids (not just step_id=0) because it doesn't substitute
    tokens — it only stores targets for an auxiliary cross-entropy loss.

    For each group with both correct and incorrect rollouts:
    1. Pick the best correct rollout as donor (highest sp_score)
    2. For each incorrect rollout, store the donor's response tokens as target

    Args:
        batch: DataProto with batch tensors and non_tensor_batch arrays.

    Returns:
        batch: Modified in-place with opd_target_responses and opd_target_mask.
        metrics: Dict of OPD target metrics.
    """
    extract_match = batch.non_tensor_batch['extract_match']
    uids = batch.non_tensor_batch['uid']
    step_ids = batch.non_tensor_batch['step_id']
    sp_scores = batch.non_tensor_batch.get('sp_scores', None)
    is_hindsight = batch.non_tensor_batch.get('is_hindsight', None)

    batch_size = len(extract_match)
    response_length = batch.batch['responses'].shape[1]

    # Initialize target tensors
    opd_target_responses = torch.zeros(batch_size, response_length, dtype=batch.batch['responses'].dtype)
    opd_target_mask = np.zeros(batch_size, dtype=np.float32)

    # Group indices by (uid, step_id)
    groups = defaultdict(list)
    for i in range(batch_size):
        key = (uids[i], step_ids[i])
        groups[key].append(i)

    n_target_samples = 0
    n_hindsight_donors = 0

    for key, indices in groups.items():
        if len(indices) <= 1:
            continue

        correct_idxs = [i for i in indices if extract_match[i]]
        incorrect_idxs = [i for i in indices if not extract_match[i]]

        if len(correct_idxs) == 0 or len(incorrect_idxs) == 0:
            continue

        # Pick best correct rollout as donor (purely by sp_score)
        # No hindsight preference — early in training the model can't leverage
        # s_{t+1} yet, so hindsight rollouts aren't necessarily better donors.
        if sp_scores is not None:
            donor_idx = max(correct_idxs, key=lambda i: sp_scores[i])
        else:
            donor_idx = correct_idxs[0]

        if is_hindsight is not None and is_hindsight[donor_idx]:
            n_hindsight_donors += 1

        donor_response = batch.batch['responses'][donor_idx]

        # Store donor's response as target for all incorrect rollouts
        # (hindsight incorrect rollouts are valid recipients for auxiliary loss)
        for r_idx in incorrect_idxs:
            opd_target_responses[r_idx] = donor_response.clone()
            opd_target_mask[r_idx] = 1.0
            n_target_samples += 1

    # Store in batch (both as tensors so they flow through select_keys)
    batch.batch['opd_target_responses'] = opd_target_responses
    batch.batch['opd_target_mask'] = torch.tensor(opd_target_mask, dtype=torch.float32)

    metrics = {
        'opd/n_target_samples': n_target_samples,
        'opd/n_hindsight_donors': n_hindsight_donors,
        'opd/target_coverage': n_target_samples / batch_size if batch_size > 0 else 0.0,
    }

    return batch, metrics


def construct_hindsight_batch(batch, msg_man, max_samples=8):
    """
    V7: Construct hindsight auxiliary loss data.

    For each (uid, step_id) group with mixed correct/incorrect rollouts and
    s_{t+1} available, construct enriched prompts (original + s_{t+1} screenshot)
    paired with the best correct donor response as CE target.

    Args:
        batch: DataProto with non_tensor_batch containing raw_messages,
               next_screenshot_path, extract_match, uid, step_id, sp_scores.
        msg_man: QwenMessages2Inputs instance (provides tokenizer, processor).

    Returns:
        hindsight_data: Dict with padded tensors for hindsight forward pass,
                        or None if no hindsight samples could be constructed.
        metrics: Dict of hindsight construction metrics.
    """
    from qwen_vl_utils import process_vision_info
    from x.qwen.data_format import slim_messages
    from x.qwen.image import make_qwen_image_item

    import verl.utils.torch_functional as verl_F
    from verl.models.transformers.qwen2_vl import get_rope_index

    extract_match = batch.non_tensor_batch['extract_match']
    uids = batch.non_tensor_batch['uid']
    step_ids = batch.non_tensor_batch['step_id']
    sp_scores = batch.non_tensor_batch.get('sp_scores', None)
    raw_messages_arr = batch.non_tensor_batch.get('raw_messages', None)
    next_screenshot_arr = batch.non_tensor_batch.get('next_screenshot_path', None)
    next_desc_t1_arr = batch.non_tensor_batch.get('next_desc_t1', None)

    if raw_messages_arr is None or next_screenshot_arr is None:
        return None, {'hindsight/n_samples': 0}

    batch_size = len(extract_match)
    response_length = batch.batch['responses'].shape[1]

    # Group by (uid, step_id)
    groups = defaultdict(list)
    for i in range(batch_size):
        key = (uids[i], step_ids[i])
        groups[key].append(i)

    hindsight_samples = []
    n_groups_with_donor = 0
    n_skipped_no_next_ss = 0
    n_skipped_error = 0

    for key, indices in groups.items():
        correct_idxs = [i for i in indices if extract_match[i]]
        incorrect_idxs = [i for i in indices if not extract_match[i]]

        if not correct_idxs or not incorrect_idxs:
            continue

        # Pick best correct donor (highest sp_score)
        if sp_scores is not None:
            donor_idx = max(correct_idxs, key=lambda i: sp_scores[i])
        else:
            donor_idx = correct_idxs[0]

        donor_response = batch.batch['responses'][donor_idx]
        donor_response_mask = batch.batch['attention_mask'][donor_idx, -response_length:]
        n_groups_with_donor += 1

        for r_idx in incorrect_idxs:
            if next_screenshot_arr[r_idx] is None:
                n_skipped_no_next_ss += 1
                continue

            try:
                raw_msgs = copy.deepcopy(raw_messages_arr[r_idx]['msgs'])
                next_ss_path = next_screenshot_arr[r_idx]

                # Append s_{t+1} screenshot to last user message
                assert raw_msgs[-1]['role'] == 'user'
                raw_msgs[-1]['content'].append({"text": "Screenshot after correct action:\n"})
                raw_msgs[-1]['content'].append(make_qwen_image_item(next_ss_path))

                # Append pi_V text description of s_{t+1} (dual-channel hindsight)
                if next_desc_t1_arr is not None:
                    desc_t1 = next_desc_t1_arr[r_idx]
                    if desc_t1:
                        raw_msgs[-1]['content'].append({
                            "text": f"\nDescription of the screen after correct action:\n{desc_t1}\n"
                        })

                # Process with num_image_limit=3 (current + s_{t+1})
                enriched_msgs = slim_messages(raw_msgs, num_image_limit=3)

                # Set min/max pixels on image elements
                for msg in enriched_msgs:
                    for content in msg['content']:
                        if 'image' in content:
                            if 'min_pixels' not in content:
                                content['min_pixels'] = msg_man.min_pixels
                            if 'max_pixels' not in content:
                                content['max_pixels'] = msg_man.max_pixels

                # Tokenize enriched prompt
                raw_prompt = msg_man.processor.apply_chat_template(
                    enriched_msgs, add_generation_prompt=True, tokenize=False)
                image_inputs, video_inputs = process_vision_info(enriched_msgs)
                model_inputs = msg_man.processor(
                    text=[raw_prompt], images=image_inputs, videos=video_inputs,
                    return_tensors="pt")

                prompt_ids = model_inputs.pop("input_ids")[0]      # (prompt_len,)
                prompt_mask = model_inputs.pop("attention_mask")[0]  # (prompt_len,)
                model_inputs.pop("second_per_grid_ts", None)
                mm_inputs = dict(model_inputs)

                # Truncate prompt if too long
                max_prompt_len = msg_man.max_prompt_length
                if prompt_ids.size(0) > max_prompt_len:
                    prompt_ids = prompt_ids[-max_prompt_len:]
                    prompt_mask = prompt_mask[-max_prompt_len:]

                hindsight_samples.append({
                    'prompt_ids': prompt_ids,
                    'prompt_mask': prompt_mask,
                    'mm_inputs': mm_inputs,
                    'donor_response': donor_response.clone(),
                    'donor_response_mask': donor_response_mask.clone(),
                })
            except Exception as e:
                n_skipped_error += 1
                print(f"[Hindsight] Error constructing sample for idx={r_idx}: {e}")
                traceback.print_exc()
                continue

    # Cap hindsight samples to avoid OOM (pixel_values are large)
    if len(hindsight_samples) > max_samples:
        import random
        random.shuffle(hindsight_samples)
        hindsight_samples = hindsight_samples[:max_samples]

    n_incorrect_total = sum(1 for i in range(batch_size) if not extract_match[i])

    if not hindsight_samples:
        return None, {
            'hindsight/n_samples': 0,
            'hindsight/coverage': 0.0,
            'hindsight/n_groups_with_donor': n_groups_with_donor,
            'hindsight/n_skipped_no_next_ss': n_skipped_no_next_ss,
            'hindsight/n_skipped_error': n_skipped_error,
        }

    # Collate: find max prompt length and pad
    max_plen = max(s['prompt_ids'].size(0) for s in hindsight_samples)
    total_len = max_plen + response_length
    n_hs = len(hindsight_samples)
    pad_token_id = msg_man.tokenizer.pad_token_id

    all_input_ids = torch.full((n_hs, total_len), pad_token_id, dtype=torch.long)
    all_attention_mask = torch.zeros((n_hs, total_len), dtype=torch.long)
    all_position_ids = torch.zeros((n_hs, 3, total_len), dtype=torch.long)
    all_responses = torch.zeros((n_hs, response_length), dtype=torch.long)
    all_response_mask = torch.zeros((n_hs, response_length), dtype=torch.float32)
    all_mm_inputs = []

    for j, s in enumerate(hindsight_samples):
        prompt_len = s['prompt_ids'].size(0)
        pad_len = max_plen - prompt_len

        # Left-pad prompt, then append donor response
        all_input_ids[j, pad_len:max_plen] = s['prompt_ids']
        all_input_ids[j, max_plen:] = s['donor_response']

        all_attention_mask[j, pad_len:max_plen] = s['prompt_mask']
        all_attention_mask[j, max_plen:] = s['donor_response_mask']

        all_responses[j] = s['donor_response']
        all_response_mask[j] = s['donor_response_mask'].float()

        # Compute Qwen2-VL RoPE position_ids for the full sequence
        pos_ids = get_rope_index(
            msg_man.processor,
            input_ids=all_input_ids[j],
            image_grid_thw=s['mm_inputs'].get('image_grid_thw'),
            video_grid_thw=s['mm_inputs'].get('video_grid_thw'),
            second_per_grid_ts=None,
            attention_mask=all_attention_mask[j],
        )  # (3, total_len)
        all_position_ids[j] = pos_ids

        all_mm_inputs.append(s['mm_inputs'])

    hindsight_data = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'position_ids': all_position_ids,
        'responses': all_responses,
        'response_mask': all_response_mask,
        'multi_modal_inputs': all_mm_inputs,
    }

    metrics = {
        'hindsight/n_samples': n_hs,
        'hindsight/coverage': n_hs / max(n_incorrect_total, 1),
        'hindsight/n_groups_with_donor': n_groups_with_donor,
        'hindsight/n_skipped_no_next_ss': n_skipped_no_next_ss,
        'hindsight/n_skipped_error': n_skipped_error,
    }

    print(f"[Hindsight] Constructed {n_hs} samples from {n_groups_with_donor} groups "
          f"(coverage={metrics['hindsight/coverage']:.2%}, "
          f"skipped_no_next_ss={n_skipped_no_next_ss}, skipped_error={n_skipped_error})")

    return hindsight_data, metrics
