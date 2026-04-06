"""
Sequential Progress (SP) reward and SPWA weighting for trajectory-level RL.

SP = first_error_step / total_steps (1.0 if all correct)
SPWA = per-step advantage weight based on marginal contribution to SP
"""

from collections import defaultdict

import numpy as np


def compute_sequential_progress(batch):
    """Compute Sequential Progress (SP) for each rollout in the batch.

    Groups steps by traj_uid, finds the first step where extract_match == False,
    and computes SP = first_error_step / total_steps.

    Args:
        batch: DataProto with non_tensor_batch containing 'traj_uid', 'step_id', 'extract_match'

    Returns:
        sp_scores: np.array (bs,) — SP score broadcast to every step of that rollout
        first_error_steps: dict {traj_uid: first_error_step_index}
    """
    traj_uids = batch.non_tensor_batch['traj_uid']
    step_ids = batch.non_tensor_batch['step_id']
    extract_matches = batch.non_tensor_batch['extract_match']

    # Group indices by traj_uid
    traj_groups = defaultdict(list)
    for i, uid in enumerate(traj_uids):
        traj_groups[uid].append(i)

    sp_by_traj = {}
    first_error_steps = {}

    for uid, indices in traj_groups.items():
        # Sort by step_id
        sorted_indices = sorted(indices, key=lambda i: step_ids[i])
        total_steps = len(sorted_indices)

        first_error = total_steps  # default: all correct
        for rank, idx in enumerate(sorted_indices):
            if not extract_matches[idx]:
                first_error = rank
                break

        sp = first_error / total_steps
        sp_by_traj[uid] = sp
        first_error_steps[uid] = first_error

    # Broadcast SP to every step
    bs = len(traj_uids)
    sp_scores = np.zeros(bs, dtype=np.float32)
    for i in range(bs):
        sp_scores[i] = sp_by_traj[traj_uids[i]]

    return sp_scores, first_error_steps


def compute_spwa_weights(sp_scores, first_error_steps, traj_uids, step_ids, decay=0.5):
    """Compute SPWA (Sequential Progress Weighted Advantage) per-step weights.

    Steps before first error: weight = 1.0
    First error step: weight = 1.0 (critical divergence point)
    Steps after first error: weight = decay^(t - first_error), min 0.1

    Args:
        sp_scores: np.array (bs,) — SP scores (unused directly, kept for API consistency)
        first_error_steps: dict {traj_uid: first_error_step_index}
        traj_uids: np.array (bs,) — trajectory UIDs
        step_ids: np.array (bs,) — step indices within each trajectory
        decay: float — decay factor for steps after first error

    Returns:
        weights: np.array (bs,) — per-step SPWA weights
    """
    bs = len(traj_uids)
    weights = np.ones(bs, dtype=np.float32)

    # Build step rank within each trajectory
    traj_groups = defaultdict(list)
    for i, uid in enumerate(traj_uids):
        traj_groups[uid].append(i)

    for uid, indices in traj_groups.items():
        sorted_indices = sorted(indices, key=lambda i: step_ids[i])
        first_error = first_error_steps.get(uid, len(sorted_indices))

        for rank, idx in enumerate(sorted_indices):
            if rank < first_error:
                weights[idx] = 1.0
            elif rank == first_error:
                weights[idx] = 1.0
            else:
                w = decay ** (rank - first_error)
                weights[idx] = max(w, 0.1)

    return weights
