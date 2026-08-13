import torch


FALLBACK = -1


def sequential_select(
    ordered_candidate_indices,
    candidate_success_probability,
    fallback_success_probability,
    budget,
    minimum_delta,
    maximum_loss_risk,
):
    if (
        ordered_candidate_indices.ndim != 2
        or candidate_success_probability.ndim != 2
        or ordered_candidate_indices.shape != candidate_success_probability.shape
        or fallback_success_probability.shape != (len(ordered_candidate_indices),)
        or ordered_candidate_indices.dtype != torch.long
        or not torch.isfinite(candidate_success_probability).all()
        or not torch.isfinite(fallback_success_probability).all()
        or torch.any(candidate_success_probability < 0)
        or torch.any(candidate_success_probability > 1)
        or torch.any(fallback_success_probability < 0)
        or torch.any(fallback_success_probability > 1)
    ):
        raise ValueError("Sequential policy input mismatch")
    candidates = ordered_candidate_indices.shape[1]
    if (
        not isinstance(budget, int)
        or not 1 <= budget <= candidates
        or minimum_delta < 0
        or not 0 <= maximum_loss_risk <= 1
    ):
        raise ValueError("Sequential policy threshold mismatch")
    if any(
        len(set(int(value) for value in row.tolist())) != candidates
        or torch.any(row < 0)
        or torch.any(row >= candidates)
        for row in ordered_candidate_indices
    ):
        raise ValueError("Sequential policy order must be a permutation")
    selected = torch.full(
        (len(ordered_candidate_indices),), FALLBACK,
        dtype=torch.long, device=ordered_candidate_indices.device,
    )
    inspected = torch.zeros_like(selected)
    accepted = torch.zeros_like(selected, dtype=torch.bool)
    for step in range(budget):
        active = ~accepted
        inspected[active] += 1
        probability = candidate_success_probability[:, step]
        baseline = fallback_success_probability
        expected_delta = probability - baseline
        loss_risk = baseline * (1.0 - probability)
        take = (
            active
            & (expected_delta >= minimum_delta)
            & (loss_risk <= maximum_loss_risk)
        )
        selected[take] = ordered_candidate_indices[take, step]
        accepted |= take
    return {
        "selected_candidate": selected,
        "accepted": accepted,
        "inspected_candidates": inspected,
        "used_fallback": ~accepted,
    }