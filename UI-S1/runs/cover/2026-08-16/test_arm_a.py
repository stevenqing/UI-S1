import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("arm_a", RUN_DIR / "arm_a.py")
arm_a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arm_a
SPEC.loader.exec_module(arm_a)


class ArmATest(unittest.TestCase):
    def test_coverage_map_half_open(self):
        values = arm_a.coverage_map(3, 3, [[0, 0, 2, 2], [1, 1, 3, 3]])
        self.assertEqual(values.tolist(), [[1, 1, 0], [1, 2, 1], [0, 1, 1]])

    def test_target_strata(self):
        self.assertEqual(arm_a.target_stratum(11), "common_11")
        self.assertEqual(arm_a.target_stratum(0), "uncovered_0")
        self.assertEqual(arm_a.target_stratum(5), "partial_1_10")

    def test_row_class(self):
        candidates = [{"correct": False}, {"correct": True}]
        self.assertEqual(arm_a.row_class(candidates, 0), "recoverable")
        self.assertEqual(arm_a.row_class(candidates, 1), "selected_correct")


if __name__ == "__main__":
    unittest.main()