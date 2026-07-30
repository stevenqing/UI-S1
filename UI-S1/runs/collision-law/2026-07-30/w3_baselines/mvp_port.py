import math
from dataclasses import dataclass


MVP_THRESHOLD_PIXELS = 14.0


@dataclass(frozen=True)
class ClusterResult:
    coordinate: tuple[float, float] | None
    member_indices: tuple[int, ...]
    cluster_sizes: tuple[int, ...]


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