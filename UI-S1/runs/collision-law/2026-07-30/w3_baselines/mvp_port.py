import math
from dataclasses import dataclass


MVP_THRESHOLD_PIXELS = 14.0


@dataclass(frozen=True)
class ClusterResult:
    coordinate: tuple[float, float] | None
    member_indices: tuple[int, ...]
    cluster_sizes: tuple[int, ...]


@dataclass(frozen=True)
class OfficialMVPResult:
    coordinate: tuple[float, float] | None
    member_indices: tuple[int, ...]
    group_sizes: tuple[int, ...]
    selected_index: int | None
    mode: str


def _pixel_distance(left, right, image_size):
    width, height = image_size
    return math.hypot((left[0] - right[0]) * width, (left[1] - right[1]) * height)


def multi_coordinate_clustering(
    points: list[tuple[float, float]],
    image_size: tuple[int, int],
    threshold_pixels: float = MVP_THRESHOLD_PIXELS,
) -> ClusterResult:
    if threshold_pixels != MVP_THRESHOLD_PIXELS:
        raise ValueError("main MVP threshold is fixed at 14 pixels")
    if not points:
        return ClusterResult(None, (), ())
    adjacency = {index: set() for index in range(len(points))}
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            if _pixel_distance(points[left], points[right], image_size) <= threshold_pixels:
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
    coordinate = (
        sum(points[index][0] for index in winner) / len(winner),
        sum(points[index][1] for index in winner) / len(winner),
    )
    return ClusterResult(
        coordinate, winner,
        tuple(sorted((len(component) for component in components), reverse=True)),
    )


def official_complete_link_groups(
    points: list[tuple[float, float]],
    threshold_pixels: float = MVP_THRESHOLD_PIXELS,
) -> list[tuple[int, ...]]:
    if threshold_pixels != MVP_THRESHOLD_PIXELS:
        raise ValueError("official MVP threshold is fixed at 14 pixels")
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
                abs(points[member][0] - points[candidate][0]) <= threshold_pixels
                and abs(points[member][1] - points[candidate][1]) <= threshold_pixels
                for member in members
            ):
                members.append(candidate)
                assigned.add(candidate)
        groups.append(tuple(members))
    groups.sort(key=len, reverse=True)
    return groups


def mvp_official_code(
    points_pixels: list[tuple[float, float]],
    coverage: list[float],
) -> OfficialMVPResult:
    if len(points_pixels) != len(coverage):
        raise ValueError("MVP points and coverage differ in length")
    if not points_pixels:
        return OfficialMVPResult(None, (), (), None, "official_code")
    groups = official_complete_link_groups(points_pixels)
    group_scores = []
    for group_index, group in enumerate(groups):
        mean_coverage = sum(coverage[index] for index in group) / len(group)
        group_scores.append((len(group) + mean_coverage / 1000, -group_index, group))
    winner = max(group_scores)[2]
    selected = max(winner, key=lambda index: (coverage[index], -index))
    return OfficialMVPResult(
        points_pixels[selected], winner, tuple(len(group) for group in groups),
        selected, "official_code",
    )


def mvp_paper_centroid(
    points_pixels: list[tuple[float, float]],
    coverage: list[float],
) -> OfficialMVPResult:
    official = mvp_official_code(points_pixels, coverage)
    if not official.member_indices:
        return OfficialMVPResult(None, (), (), None, "paper_centroid")
    centroid = (
        sum(points_pixels[index][0] for index in official.member_indices) / len(official.member_indices),
        sum(points_pixels[index][1] for index in official.member_indices) / len(official.member_indices),
    )
    return OfficialMVPResult(
        centroid, official.member_indices, official.group_sizes,
        None, "paper_centroid",
    )