import math
from functools import lru_cache

from scoring import token_f1


ANDROID_GROUNDING_RADIUS = 0.14
ANDROID_GAUSSIAN_SIGMA = ANDROID_GROUNDING_RADIUS / 2
ANDROID_RHO0 = 0.30712361963678886
UNIT_SQUARE_DIAMETER = math.sqrt(2.0)


def assert_main_kernel_constants() -> None:
    assert ANDROID_GROUNDING_RADIUS == 0.14
    assert ANDROID_GAUSSIAN_SIGMA == 0.07
    assert math.isclose(ANDROID_RHO0, 0.30712361963678886, rel_tol=0, abs_tol=1e-16)
    assert math.isclose(UNIT_SQUARE_DIAMETER, math.sqrt(2.0), rel_tol=0, abs_tol=0)


def type_kernel(left: str, right: str) -> float:
    return float(left == right)


def android_coord_kernel(left: tuple[float, float], right: tuple[float, float]) -> float:
    distance = math.dist(left, right)
    return math.exp(-(distance * distance) / (2 * ANDROID_GAUSSIAN_SIGMA**2))


def android_coord_kernel_normalized(left: tuple[float, float], right: tuple[float, float]) -> float:
    return android_coord_kernel(left, right) / ANDROID_RHO0


def android_coord_hard_ablation(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(math.dist(left, right) < ANDROID_GROUNDING_RADIUS)


def mind2web_coord_inference(left: tuple[float, float], right: tuple[float, float]) -> float:
    for point in (left, right):
        if len(point) != 2 or not all(0.0 <= value <= 1.0 for value in point):
            raise ValueError(f"Mind2Web inference point outside unit square: {point}")
    return max(0.0, 1.0 - math.dist(left, right) / UNIT_SQUARE_DIAMETER)


def mind2web_point_in_bbox_gt_analysis(
    point: tuple[float, float],
    gt_bbox: tuple[float, float, float, float],
) -> float:
    x, y = point
    x0, y0, x1, y1 = gt_bbox
    return float(x0 <= x <= x1 and y0 <= y <= y1)


def mind2web_signed_bbox_distance_gt_analysis(
    point: tuple[float, float],
    gt_bbox: tuple[float, float, float, float],
) -> float:
    x, y = point
    x0, y0, x1, y1 = gt_bbox
    if x0 <= x <= x1 and y0 <= y <= y1:
        return min(x - x0, x1 - x, y - y0, y1 - y)
    delta_x = max(x0 - x, 0.0, x - x1)
    delta_y = max(y0 - y, 0.0, y - y1)
    return -math.hypot(delta_x, delta_y)


@lru_cache(maxsize=65536)
def string_kernel(left: str, right: str) -> float:
    return token_f1(left.lower(), right.lower())


assert_main_kernel_constants()