import math

import numpy as np


MVP_THRESHOLD_PIXELS = 14.0


def full_image(points, candidates):
    return points[0]


def random_candidate(points, candidates, random_index):
    return points[random_index]


def coordinate_mean(points, candidates):
    return tuple(np.mean(np.asarray(points, dtype=np.float64), axis=0).tolist())


def official_groups(points):
    groups = []
    assigned = set()
    for index in range(len(points)):
        if index in assigned:
            continue
        members = [index]
        assigned.add(index)
        for candidate in range(len(points)):
            if candidate in assigned:
                continue
            if all(
                abs(points[member][0] - points[candidate][0]) <= MVP_THRESHOLD_PIXELS
                and abs(points[member][1] - points[candidate][1]) <= MVP_THRESHOLD_PIXELS
                for member in members
            ):
                members.append(candidate)
                assigned.add(candidate)
        groups.append(tuple(members))
    groups.sort(key=len, reverse=True)
    return groups


def mvp_official(points, candidates):
    groups = official_groups(points)
    scored = []
    for group_index, group in enumerate(groups):
        coverage = sum(candidates[index]["coverage"] for index in group) / len(group)
        scored.append((len(group) + coverage / 1000, -group_index, group))
    winner = max(scored)[2]
    selected = max(winner, key=lambda index: (candidates[index]["coverage"], -index))
    return points[selected]


def mvp_paper_centroid(points, candidates):
    group = official_groups(points)[0]
    return tuple(np.mean(np.asarray([points[index] for index in group]), axis=0).tolist())


def mvp_graph_centroid(points, candidates):
    adjacency = {index: set() for index in range(len(points))}
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            if math.dist(points[left], points[right]) <= MVP_THRESHOLD_PIXELS:
                adjacency[left].add(right)
                adjacency[right].add(left)
    components = []
    unvisited = set(range(len(points)))
    while unvisited:
        seed = min(unvisited)
        stack = [seed]
        component = []
        unvisited.remove(seed)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    winner = max(components, key=lambda component: (len(component), -min(component)))
    return tuple(np.mean(np.asarray([points[index] for index in winner]), axis=0).tolist())


def scott_bandwidth(points):
    if len(points) <= 1:
        return 1e-6
    values = np.asarray(points, dtype=np.float64)
    scale = float(np.sqrt(np.mean(np.var(values, axis=0, ddof=1))))
    return max(scale * len(points) ** (-1 / 6), 1e-6)


def kde_peak(points):
    bandwidth = scott_bandwidth(points)
    scores = [
        sum(math.exp(-(math.dist(point, candidate) ** 2) / (2 * bandwidth**2)) for point in points)
        for candidate in points
    ]
    return max(range(len(points)), key=lambda index: (scores[index], -index))


def reguide_algorithm_level(points, candidates):
    first = kde_peak(points)
    first_point = points[first]
    retained = []
    retained_points = []
    for index, candidate in enumerate(candidates):
        if index == 0:
            continue
        left, top, right, bottom = candidate["region"]
        if left <= first_point[0] <= right and top <= first_point[1] <= bottom:
            retained.append(index)
            retained_points.append(points[index])
    if not retained:
        return first_point
    second = kde_peak(retained_points)
    return points[retained[second]]
