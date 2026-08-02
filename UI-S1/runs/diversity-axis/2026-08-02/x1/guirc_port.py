import numpy as np
from scipy.ndimage import find_objects, label


POINT_EXPAND_SIZE = 50


def point_to_box(point, width, height, point_expand_size=POINT_EXPAND_SIZE):
    if point is None:
        return [0, 0, 0, 0]
    expand = point_expand_size / 2
    x, y = point
    return [
        max(0, int(x - expand)),
        max(0, int(y - expand)),
        min(width, int(x + expand)),
        min(height, int(y + expand)),
    ]


def region_consistency_vote(points, width, height, point_expand_size=POINT_EXPAND_SIZE):
    boxes = [point_to_box(point, width, height, point_expand_size) for point in points]
    valid = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    if not valid:
        return {
            "point": [0.0, 0.0],
            "consensus_region": None,
            "max_votes": 0,
            "sampled_boxes": boxes,
        }

    offset_x = min(box[0] for box in valid)
    offset_y = min(box[1] for box in valid)
    right = max(box[2] for box in valid)
    bottom = max(box[3] for box in valid)
    grid = np.zeros((bottom - offset_y, right - offset_x), dtype=np.int32)
    for x1, y1, x2, y2 in valid:
        grid[y1 - offset_y:y2 - offset_y, x1 - offset_x:x2 - offset_x] += 1

    max_votes = int(grid.max())
    labeled, count = label(grid == max_votes)
    if count <= 0:
        raise AssertionError("positive vote grid must have a connected component")
    max_area = 0
    selected = None
    for region_id, slices in enumerate(find_objects(labeled)):
        if slices is None:
            continue
        y_slice, x_slice = slices
        area = int(np.sum(labeled == region_id + 1))
        if area > max_area:
            max_area = area
            selected = [
                x_slice.start + offset_x,
                y_slice.start + offset_y,
                x_slice.stop + offset_x,
                y_slice.stop + offset_y,
            ]
    if selected is None:
        raise AssertionError("GUI-RC failed to select a max-vote component")
    return {
        "point": [(selected[0] + selected[2]) / 2, (selected[1] + selected[3]) / 2],
        "consensus_region": selected,
        "max_votes": max_votes,
        "sampled_boxes": boxes,
    }
