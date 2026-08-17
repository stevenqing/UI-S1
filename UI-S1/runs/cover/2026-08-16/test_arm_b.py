import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("arm_b", RUN_DIR / "arm_b.py")
arm_b = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arm_b
SPEC.loader.exec_module(arm_b)


class Candidate:
    def __init__(self, lineage):
        self.lineage = lineage


class ArmBTest(unittest.TestCase):
    def test_pair_counts(self):
        candidates = [Candidate(model) for _ in range(4) for model in ("a", "b", "c")]
        self.assertEqual(sum(arm_b.pair_primary_stratum(candidates, *pair) == "within_model" for pair in arm_b.PAIRS), 18)
        self.assertEqual(sum(arm_b.pair_primary_stratum(candidates, *pair) == "cross_model" for pair in arm_b.PAIRS), 48)
        counts = {name: sum(arm_b.pair_trend_stratum(candidates, *pair) == name for pair in arm_b.PAIRS) for name in arm_b.TREND_ORDER}
        self.assertEqual(counts, {"within_model_cross_slot": 18, "cross_model_matched_role": 12, "cross_model_unmatched_role": 36})

    def test_ordering(self):
        values = {"within_model_cross_slot": 0.8, "cross_model_matched_role": 0.6, "cross_model_unmatched_role": 0.4}
        self.assertEqual(arm_b.ordering(values), list(arm_b.TREND_ORDER))


if __name__ == "__main__":
    unittest.main()