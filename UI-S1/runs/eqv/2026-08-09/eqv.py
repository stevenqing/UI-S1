import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    action: str
    coordinate: tuple[float, float] | None
    parameter: str
    lineage: str
    source: str
    reliability: float
    order: int
    payload: object = None
    parse_ok: bool = True


def token_set_f1(left, right):
    left_tokens = set(str(left or "").lower().split())
    right_tokens = set(str(right or "").lower().split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def coordinate_equivalent(left, right, threshold, metric):
    if left is None or right is None:
        return left is None and right is None
    if metric == "euclidean":
        return math.dist(left, right) <= float(threshold)
    if metric == "axis_aligned":
        width, height = threshold
        return abs(left[0] - right[0]) <= width and abs(left[1] - right[1]) <= height
    raise ValueError(f"unknown coordinate metric: {metric}")


def equivalent(left, right, *, use_action, use_coordinate, use_parameter, coordinate_threshold, coordinate_metric, parameter_threshold):
    if use_action and left.action != right.action:
        return False
    if use_coordinate and not coordinate_equivalent(left.coordinate, right.coordinate, coordinate_threshold, coordinate_metric):
        return False
    if use_parameter and token_set_f1(left.parameter, right.parameter) < parameter_threshold:
        return False
    return True


def complete_link_classes(candidates, equivalence):
    classes = []
    for index, candidate in enumerate(candidates):
        for members in classes:
            if all(equivalence(candidate, candidates[member]) for member in members):
                members.append(index)
                break
        else:
            classes.append([index])
    return tuple(tuple(members) for members in classes)


def single_link_classes(candidates, equivalence):
    adjacency = {index: set() for index in range(len(candidates))}
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if equivalence(candidates[left], candidates[right]):
                adjacency[left].add(right)
                adjacency[right].add(left)
    classes = []
    unvisited = set(adjacency)
    while unvisited:
        seed = min(unvisited)
        stack = [seed]
        unvisited.remove(seed)
        members = []
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
        classes.append(tuple(sorted(members)))
    return tuple(classes)


def retained_members(candidates, members, lineage_dedup):
    if not lineage_dedup:
        return tuple(members)
    by_lineage = {}
    for index in members:
        candidate = candidates[index]
        current = by_lineage.get(candidate.lineage)
        if current is None or (candidate.reliability, -candidate.order) > (candidates[current].reliability, -candidates[current].order):
            by_lineage[candidate.lineage] = index
    return tuple(sorted(by_lineage.values()))


def aggregate(candidates, *, equivalence, linkage="complete", lineage_dedup=True, lineage_order=()):
    parsed = [candidate for candidate in candidates if candidate.parse_ok]
    if not parsed:
        return {"prediction": None, "classes": (), "winning_class": (), "retained_members": (), "votes": 0}
    classes = complete_link_classes(parsed, equivalence) if linkage == "complete" else single_link_classes(parsed, equivalence)
    lineage_rank = {lineage: index for index, lineage in enumerate(lineage_order)}
    scored = []
    for class_index, members in enumerate(classes):
        retained = retained_members(parsed, members, lineage_dedup)
        earliest_lineage = min((lineage_rank.get(parsed[index].lineage, len(lineage_rank)) for index in retained), default=len(lineage_rank))
        score = (
            len(retained),
            sum(parsed[index].reliability for index in retained),
            max((parsed[index].reliability for index in retained), default=0.0),
            -earliest_lineage,
            -min(parsed[index].order for index in retained),
            -class_index,
        )
        scored.append((score, members, retained))
    _, winning_class, retained = max(scored, key=lambda item: item[0])
    selected = max(retained, key=lambda index: (parsed[index].reliability, -lineage_rank.get(parsed[index].lineage, len(lineage_rank)), -parsed[index].order))
    return {
        "prediction": parsed[selected],
        "classes": classes,
        "winning_class": winning_class,
        "retained_members": retained,
        "votes": len(retained),
    }


def contract_tests():
    candidates = [
        Candidate("CLICK", (0.0, 0.0), "", "A", "A0", 0.4, 0),
        Candidate("CLICK", (0.1, 0.0), "", "A", "A1", 0.8, 1),
        Candidate("CLICK", (0.2, 0.0), "", "B", "B0", 0.7, 2),
    ]
    relation = lambda left, right: coordinate_equivalent(left.coordinate, right.coordinate, 0.11, "euclidean")
    assert complete_link_classes(candidates, relation) == ((0, 1), (2,))
    assert single_link_classes(candidates, relation) == ((0, 1, 2),)
    result = aggregate(candidates[:2], equivalence=relation, lineage_order=("A",))
    assert result["votes"] == 1 and result["prediction"].source == "A1"
    assert math.isclose(token_set_f1("hello world", "hello"), 2 / 3)


if __name__ == "__main__":
    contract_tests()
    print("PASS")
