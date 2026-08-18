import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import owin_common as common


class OwinGeometryTest(unittest.TestCase):
    def test_amendment_001_offsets(self):
        self.assertEqual(common.jitter_offsets(109.2), [[0, 0], [9, 6], [9, 20], [-5, 32], [-29, 33], [-52, 15], [-63, -18], [-50, -58], [-12, -86], [41, -89], [92, -59]])

    def test_oracle_window_repairs_and_contains_center(self):
        result = common.oracle_window(3840, 2160, [3800, 2100, 3840, 2160], [92, -59])
        self.assertEqual(result["final_window"], [2552, 1432, 3840, 2160])
        self.assertTrue(result["target_center_contained"])

    def test_union_area(self):
        self.assertEqual(common.union_area([[0, 0, 2, 2], [1, 0, 3, 2]]), 6)

    def test_center_and_full_bbox_are_distinct(self):
        rectangle = [0, 0, 10, 10]
        target_bbox = [-1, 4, 9, 6]
        self.assertTrue(common.contains_center(rectangle, target_bbox))
        self.assertFalse(common.contains_bbox(rectangle, target_bbox))

    def test_uniform_anchors(self):
        self.assertEqual(common.uniform_anchors(5, 3), [0, 3, 5])
        self.assertEqual(common.uniform_anchors(5, 1), [3])

    def test_tiling_is_deterministic_and_in_bounds(self):
        first = common.tiling_layout(3840, 2160, 11)
        second = common.tiling_layout(3840, 2160, 11)
        self.assertEqual(first, second)
        self.assertEqual(len(first["rectangles"]), 11)
        self.assertTrue(all(0 <= left < right <= 3840 and 0 <= top < bottom <= 2160 for left, top, right, bottom in first["rectangles"]))


if __name__ == "__main__":
    unittest.main()