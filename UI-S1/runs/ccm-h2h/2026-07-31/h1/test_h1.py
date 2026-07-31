import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from aggregators_coord import mvp_official, reguide_algorithm_level
from ccm_coord import fit, select
from merge_candidates import candidate_hash, main as merge_main
from generation_contract import generation_row


class H1HeadToHeadTest(unittest.TestCase):
    def test_official_mvp_selects_highest_coverage_in_largest_group(self):
        points = [(100, 100), (105, 105), (900, 700)]
        candidates = [
            {"coverage": 0, "region": [0, 0, 1000, 800]},
            {"coverage": 8, "region": [0, 0, 500, 500]},
            {"coverage": 100, "region": [500, 400, 1000, 800]},
        ]
        self.assertEqual(mvp_official(points, candidates), (105, 105))

    def test_coordinate_ccm_prefers_truth_conditional_agreement_pattern(self):
        calibration_rows = []
        for index in range(80):
            if index < 40:
                points = [(100, 100), (102, 101), (800, 700)]
                labels = [True, True, False]
            else:
                points = [(100, 100), (500, 500), (800, 700)]
                labels = [True, False, False]
            calibration_rows.append((points, labels))
        calibration = fit(calibration_rows)
        winner, scores = select(calibration, [(100, 100), (102, 101), (800, 700)])
        self.assertIn(winner, (0, 1))
        self.assertGreater(scores[winner], scores[2])

    def test_reguide_second_stage_uses_roi_candidates(self):
        points = [(100, 100), (102, 101), (103, 99), (900, 700)]
        candidates = [
            {"coverage": 0, "region": [0, 0, 1000, 800]},
            {"coverage": 8, "region": [50, 50, 200, 200]},
            {"coverage": 7, "region": [50, 50, 200, 200]},
            {"coverage": 100, "region": [800, 600, 1000, 800]},
        ]
        self.assertIn(reguide_algorithm_level(points, candidates), ((102, 101), (103, 99)))

    def test_candidate_hash_is_order_sensitive(self):
        left = [{"point": [1, 2]}, {"point": [3, 4]}]
        right = list(reversed(left))
        self.assertNotEqual(candidate_hash(left), candidate_hash(right))

    def test_generation_row_removes_target_bbox(self):
        source = {"id": "sample", "bbox": [10, 20, 30, 40], "instruction": "click"}
        sanitized = generation_row(source)
        self.assertEqual(sanitized["bbox"], [-1, -1, -1, -1])
        self.assertEqual(source["bbox"], [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
