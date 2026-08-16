import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("stage0", RUN_DIR / "stage0.py")
stage0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage0
SPEC.loader.exec_module(stage0)


class Stage0Test(unittest.TestCase):
    def test_rho_zero_is_block_size(self):
        group = (0, 1, 3)
        weights = {model: 1.0 for model in stage0.MODELS}
        self.assertEqual(stage0.effective_score(group, 0, 0, weights), 3.0)

    def test_within_lineage_saturation(self):
        weights = {model: 1.0 for model in stage0.MODELS}
        self.assertLess(stage0.effective_score((0, 3), 0.895, 0.398, weights), 2.0)

    def test_representative_uses_coverage_then_index(self):
        candidates = [{"coverage": 1}, {"coverage": 2}, {"coverage": 2}]
        self.assertEqual(stage0.representative(candidates, (0, 1, 2)), 1)

    def test_transition_pairs_separate_counts(self):
        features = [
            {"budget": 3, "lineage_count": 1, "view_count": 2},
            {"budget": 3, "lineage_count": 2, "view_count": 2},
            {"budget": 3, "lineage_count": 3, "view_count": 2},
        ]
        self.assertEqual(len(stage0.transition_pairs(features, (1, 2))), 1)
        self.assertEqual(len(stage0.transition_pairs(features, (2, 3))), 1)


if __name__ == "__main__":
    unittest.main()