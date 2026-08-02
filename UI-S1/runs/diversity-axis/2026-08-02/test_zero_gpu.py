import unittest

from x7_safeground_port import compute_uncertainty, region_scores


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


if __name__ == "__main__":
    unittest.main()