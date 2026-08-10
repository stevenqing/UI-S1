from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    action: str
    coordinate: tuple[float, float] | None
    parameter: str
    source: str
    reliability: float
    order: int
    payload: object = None
    parse_ok: bool = True
    lineage: str = ""


def token_set_f1(left, right):
    left_tokens = set(str(left or "").lower().split())
    right_tokens = set(str(right or "").lower().split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def equivalent(left, right, granularity, coordinate_threshold, parameter_threshold):
    if granularity in {"G0", "G1", "G2", "G3"} and left.action != right.action:
        return False
    if granularity in {"G1", "G3", "G4"}:
        if left.coordinate is None or right.coordinate is None:
            return left.coordinate is None and right.coordinate is None
        if isinstance(coordinate_threshold, tuple):
            if abs(left.coordinate[0] - right.coordinate[0]) > coordinate_threshold[0]:
                return False
            if abs(left.coordinate[1] - right.coordinate[1]) > coordinate_threshold[1]:
                return False
        else:
            delta_x = left.coordinate[0] - right.coordinate[0]
            delta_y = left.coordinate[1] - right.coordinate[1]
            if delta_x * delta_x + delta_y * delta_y > float(coordinate_threshold) ** 2:
                return False
    if granularity in {"G2", "G3"} and token_set_f1(left.parameter, right.parameter) < parameter_threshold:
        return False
    return True


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


def single_link_classes(candidates, relation):
    adjacency = {index: set() for index in range(len(candidates))}
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if relation(candidates[left], candidates[right]):
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


def capped_members(candidates, members, lineage_vote_cap):
    if lineage_vote_cap is None:
        return tuple(members)
    by_lineage = {}
    for index in members:
        lineage = candidates[index].lineage or candidates[index].source
        by_lineage.setdefault(lineage, []).append(index)
    retained = []
    for lineage in sorted(by_lineage):
        ranked = sorted(by_lineage[lineage], key=lambda index: (-candidates[index].reliability, candidates[index].order))
        retained.extend(ranked[:lineage_vote_cap])
    return tuple(sorted(retained))


def select(candidates, granularity, coordinate_threshold, parameter_threshold=1.0, linkage="complete", lineage_vote_cap=None):
    parsed = [candidate for candidate in candidates if candidate.parse_ok]
    if not parsed:
        return None, {"classes": [], "winning_class": [], "votes": 0}
    relation = lambda left, right: equivalent(left, right, granularity, coordinate_threshold, parameter_threshold)
    classes = complete_link_classes(parsed, relation) if linkage == "complete" else single_link_classes(parsed, relation)
    scored = []
    for members in classes:
        retained = capped_members(parsed, members, lineage_vote_cap)
        scored.append((
            (
                len(retained),
                sum(parsed[index].reliability for index in retained),
                max(parsed[index].reliability for index in retained),
                -min(parsed[index].order for index in retained),
            ),
            members,
            retained,
        ))
    _, winner, retained = max(scored, key=lambda item: item[0])
    selected = max(retained, key=lambda index: (parsed[index].reliability, -parsed[index].order))
    return parsed[selected], {
        "classes": [list(members) for members in classes],
        "winning_class": list(winner),
        "retained_members": list(retained),
        "votes": len(retained),
        "raw_class_members": len(winner),
        "linkage": linkage,
        "lineage_vote_cap": lineage_vote_cap,
    }


def contract_tests():
    candidates = [
        Candidate("POINT", (0.0, 0.0), "", "a", 0.1, 0),
        Candidate("POINT", (1.0, 0.0), "", "b", 0.9, 1),
        Candidate("POINT", (3.0, 0.0), "", "c", 0.8, 2),
    ]
    prediction, details = select(candidates, "G4", 1.1)
    assert details["winning_class"] == [0, 1]
    assert prediction.source == "b"
    assert token_set_f1("hello world", "hello") == 2 / 3


if __name__ == "__main__":
    contract_tests()
    print("PASS")
