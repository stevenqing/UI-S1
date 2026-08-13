import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class GranCandidate:
    source: str
    lineage: str
    action: str
    coordinate: tuple[float, float] | None
    parameter: str
    parse_ok: bool
    order: int
    correct: bool
    reliability: float = 0.0


def complete_link_classes(candidates, relation):
    classes = []
    for index, candidate in enumerate(candidates):
        for members in classes:
            if all(relation(candidate, candidates[member]) for member in members):
                members.append(index)
                break
        else:
            classes.append([index])
    return tuple(tuple(members) for members in classes)


def equivalent(left, right, benchmark, tau_kind, tau_value=None):
    if benchmark == "mind2web" and left.action != right.action:
        return False
    if tau_kind == "single":
        return True
    if left.coordinate is None or right.coordinate is None:
        return left.coordinate is None and right.coordinate is None
    distance = math.dist(left.coordinate, right.coordinate)
    if tau_kind == "exact":
        return distance == 0.0
    if tau_kind != "finite" or tau_value is None or tau_value <= 0:
        raise ValueError("GRAN invalid finite tau")
    return distance <= float(tau_value)


def partition(candidates, benchmark, tau_kind, tau_value=None):
    parsed = tuple(candidate for candidate in candidates if candidate.parse_ok)
    if not parsed:
        return parsed, ()
    relation = lambda left, right: equivalent(
        left, right, benchmark, tau_kind, tau_value
    )
    return parsed, complete_link_classes(parsed, relation)


def source_reliability(rows, row_ids):
    values = {}
    for row_id in row_ids:
        for candidate in rows[row_id]["candidates"]:
            values.setdefault(candidate.source, []).append(float(candidate.correct))
    if not values:
        raise ValueError("GRAN reliability fit is empty")
    return {source: float(np.mean(correct)) for source, correct in values.items()}


def attach_reliability(candidates, reliability):
    if any(candidate.source not in reliability for candidate in candidates):
        raise ValueError("GRAN candidate source absent from reliability fit")
    return tuple(
        replace(candidate, reliability=float(reliability[candidate.source]))
        for candidate in candidates
    )


def prior_select(candidates):
    parsed = [candidate for candidate in candidates if candidate.parse_ok]
    if not parsed:
        return None
    return min(parsed, key=lambda candidate: (
        -candidate.reliability, candidate.source, candidate.order
    ))


def density_select(candidates, benchmark, tau_kind, tau_value=None):
    parsed, classes = partition(candidates, benchmark, tau_kind, tau_value)
    if not parsed:
        return None, {"classes": [], "winning_class": [], "votes": 0}
    scored = []
    for members in classes:
        score = (
            len(members),
            sum(parsed[index].reliability for index in members),
            max(parsed[index].reliability for index in members),
            -min(parsed[index].order for index in members),
        )
        scored.append((score, members))
    _, winner = max(scored, key=lambda item: item[0])
    selected_index = min(winner, key=lambda index: (
        -parsed[index].reliability,
        parsed[index].source,
        parsed[index].order,
    ))
    return parsed[selected_index], {
        "classes": [list(members) for members in classes],
        "winning_class": list(winner),
        "votes": len(winner),
    }


def mechanism_values(candidates, benchmark, tau_kind, tau_value=None):
    parsed, classes = partition(candidates, benchmark, tau_kind, tau_value)
    if not parsed:
        raise ValueError("GRAN mechanism row has no parsed candidates")
    correct = np.asarray([candidate.correct for candidate in parsed], dtype=np.bool_)
    p_hat = float(np.mean(correct))
    wrong_indices = {index for index, value in enumerate(correct) if not value}
    wrong_counts = [
        sum(index in wrong_indices for index in members)
        for members in classes
    ]
    wrong_total = len(wrong_indices)
    q_max = (
        float(max(wrong_counts) / wrong_total)
        if wrong_total else 0.0
    )
    contaminated = 0
    correct_block_members = 0
    for members in classes:
        has_correct = any(correct[index] for index in members)
        if has_correct:
            correct_block_members += len(members)
            contaminated += sum(not correct[index] for index in members)
    contamination = (
        float(contaminated / correct_block_members)
        if correct_block_members else 0.0
    )
    correct_points = np.asarray([
        candidate.coordinate for candidate in parsed
        if candidate.correct and candidate.coordinate is not None
    ], dtype=np.float64)
    wrong_points = np.asarray([
        candidate.coordinate for candidate in parsed
        if not candidate.correct and candidate.coordinate is not None
    ], dtype=np.float64)
    if len(correct_points):
        center = correct_points.mean(axis=0)
        sigma_c = float(np.sqrt(np.mean(np.sum((correct_points - center) ** 2, axis=1))))
    else:
        sigma_c = None
    if len(correct_points) and len(wrong_points):
        d_min = float(min(
            math.dist(left, right)
            for left in correct_points for right in wrong_points
        ))
    else:
        d_min = None
    gamma = (
        d_min / sigma_c
        if d_min is not None and sigma_c not in (None, 0.0)
        else None
    )
    return {
        "p_hat": p_hat,
        "q_max_hat": q_max,
        "contamination": contamination,
        "alpha_hat": 1.0 - contamination,
        "sigma_c_hat": sigma_c,
        "d_min_hat": d_min,
        "gamma_hat": gamma,
        "class_count": len(classes),
        "maximum_class_share": max(len(members) for members in classes) / len(parsed),
    }


def tau_options(finite_values):
    values = [("exact", None)]
    values.extend(("finite", float(value)) for value in finite_values)
    values.append(("single", None))
    return tuple(values)


def tau_label(option):
    kind, value = option
    return kind if kind != "finite" else f"finite:{value:.17g}"