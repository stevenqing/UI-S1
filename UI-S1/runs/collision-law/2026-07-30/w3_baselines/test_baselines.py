import unittest

from mvp_port import mvp_official_code, mvp_paper_centroid, multi_coordinate_clustering, official_complete_link_groups
from pka import Prediction
from reguide_port import kde_candidate_peak, reguide_two_stage
from selfconsistency import self_consistency_product_space


class BaselineAlgorithmTest(unittest.TestCase):
    def test_mvp_largest_cluster_centroid(self):
        points = [(0.10, 0.10), (0.11, 0.10), (0.90, 0.90)]
        result = multi_coordinate_clustering(points, (1000, 1000))
        self.assertEqual(result.member_indices, (0, 1))
        self.assertAlmostEqual(result.coordinate[0], 0.105)

    def test_mvp_tie_uses_earliest_view(self):
        points = [(0.1, 0.1), (0.9, 0.9)]
        result = multi_coordinate_clustering(points, (1000, 1000))
        self.assertEqual(result.member_indices, (0,))

    def test_official_mvp_uses_axis_aligned_complete_link(self):
        points = [(0, 0), (10, 10), (20, 20)]
        self.assertEqual(official_complete_link_groups(points), [(0, 1), (2,)])

    def test_official_mvp_selects_highest_coverage_member(self):
        points = [(100, 100), (105, 105), (900, 900)]
        result = mvp_official_code(points, [0, 8, 100])
        self.assertEqual(result.member_indices, (0, 1))
        self.assertEqual(result.coordinate, (105, 105))
        centroid = mvp_paper_centroid(points, [0, 8, 100])
        self.assertEqual(centroid.coordinate, (102.5, 102.5))

    def test_kde_peak_selects_dense_candidate(self):
        result = kde_candidate_peak([(0.1, 0.1), (0.11, 0.1), (0.9, 0.9)])
        self.assertIn(result.candidate_index, (0, 1))

    def test_reguide_falls_back_to_first_stage(self):
        result = reguide_two_stage([(0.2, 0.3)], [])
        self.assertEqual(result["prediction"], (0.2, 0.3))

    def test_self_consistency_product_space(self):
        predictions = [
            Prediction("CLICK", 0.1, 0.1, source="s0"),
            Prediction("CLICK", 0.11, 0.1, source="s1"),
            Prediction("TYPE", 0.9, 0.9, "x", source="s2"),
        ]
        result = self_consistency_product_space(predictions)
        self.assertEqual(result.action, "CLICK")
        self.assertIn(result.coordinate, ((0.1, 0.1), (0.11, 0.1)))


if __name__ == "__main__":
    unittest.main()