import unittest

from x7_safeground_port import compute_uncertainty, region_scores
from x2.zoom_port import adaptive_crop, box_iou, deterministic_seed, gate, point_to_box
from x2.x2_composability import interaction_classification
from x6_unlabeled_ranking import fit_ols, mean_pairwise_normalized_distance


class SafeGroundPortTest(unittest.TestCase):
    def test_single_region_edge_values(self):
        result = compute_uncertainty([(10, 10)], 280, 280)
        self.assertEqual(result["region_scores"], [1.0])
        self.assertAlmostEqual(result["combined"], 0.18)

    def test_equal_disconnected_regions(self):
        result = compute_uncertainty([(10, 10), (250, 250)], 280, 280)
        self.assertEqual(result["region_scores"], [0.5, 0.5])
        self.assertAlmostEqual(result["combined"], 0.7, places=7)

    def test_adjacent_patches_are_one_region(self):
        result = compute_uncertainty([(10, 10), (35, 10)], 280, 280)
        self.assertEqual(result["region_scores"], [0.5])
        self.assertAlmostEqual(result["combined"], 0.26)

    def test_paper_threshold_is_strict(self):
        points = [(10, 10)] * 4 + [(250, 250)]
        self.assertEqual(region_scores(points, 280, 280, patch_size=14, activation_threshold=0.3), [0.8])


class ZoomPortTest(unittest.TestCase):
    def test_point_box_and_iou(self):
        box = point_to_box([100, 100], 200, 200)
        self.assertEqual(box, [0.375, 0.375, 0.625, 0.625])
        self.assertEqual(box_iou(box, box), 1.0)

    def test_gate_is_strict(self):
        candidates = [{"box": [0, 0, 0.5, 0.5], "confidence": 0.5}] * 3
        result = gate(candidates)
        self.assertEqual(result["score"], 1.5)
        self.assertFalse(result["reliable"])

    def test_crop_minimum_and_bounds(self):
        candidates = [
            {"box": point_to_box([10, 10], 1000, 800), "confidence": 0.5},
            {"box": point_to_box([20, 20], 1000, 800), "confidence": 0.5},
            {"box": point_to_box([900, 700], 1000, 800), "confidence": 0.5},
        ]
        crop = adaptive_crop(candidates, 1000, 800)
        self.assertEqual(crop[0:2], [0, 0])
        self.assertEqual(crop[2] - crop[0], 512)
        self.assertEqual(crop[3] - crop[1], 512)

    def test_seed_is_stable_and_slot_specific(self):
        first = deterministic_seed("row", "Q2", "GTA1", 0, 0)
        self.assertEqual(first, deterministic_seed("row", "Q2", "GTA1", 0, 0))
        self.assertNotEqual(first, deterministic_seed("row", "Q2", "GTA1", 0, 1))

    def test_interaction_classifications(self):
        self.assertEqual(interaction_classification([-0.01, 0.02]), "NEAR_ADDITIVE")
        self.assertEqual(interaction_classification([0.01, 0.02]), "SUPER_ADDITIVE")
        self.assertEqual(interaction_classification([-0.03, -0.01]), "SUB_ADDITIVE")


class PoolRankingTest(unittest.TestCase):
    def test_normalized_distance_and_ols(self):
        rows = [{
            "id": "row",
            "application": "app",
            "candidates": [{"point": [index, 0]} for index in range(12)],
        }]
        expected = sum(
            abs(left - right) / 5
            for left in range(12)
            for right in range(left + 1, 12)
        ) / 66
        self.assertAlmostEqual(
            mean_pairwise_normalized_distance(rows, {"row": [3, 4]}), expected
        )
        records = [
            {"feature": value, "target": 0.2 + 3 * value}
            for value in (0.1, 0.2, 0.3, 0.4)
        ]
        model = fit_ols(records)
        self.assertAlmostEqual(model["intercept"], 0.2)
        self.assertAlmostEqual(model["coefficient"], 3.0)
        self.assertAlmostEqual(model["training_r_squared"], 1.0)


if __name__ == "__main__":
    unittest.main()