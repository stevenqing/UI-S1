import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("arm_c", RUN_DIR / "arm_c.py")
arm_c = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arm_c
SPEC.loader.exec_module(arm_c)


class ArmCTest(unittest.TestCase):
    def test_phi_perfect(self):
        self.assertEqual(arm_c.phi_from_counts(4, 2, 2, 2), 1.0)

    def test_phi_undefined_constant(self):
        self.assertIsNone(arm_c.phi_from_counts(4, 0, 2, 0))

    def test_kappa_perfect(self):
        self.assertEqual(arm_c.kappa_from_counts(4, 2, 2, 2), 1.0)

    def test_pair_counts(self):
        self.assertEqual(sum(arm_c.pair_stratum(*pair) == "within_lineage" for pair in arm_c.PAIRS), 18)
        self.assertEqual(sum(arm_c.pair_stratum(*pair) == "cross_lineage" for pair in arm_c.PAIRS), 48)

    def test_structured_identity(self):
        self.assertEqual(arm_c.structured_neff(0, 0), 12.0)

    def test_neff_identity(self):
        self.assertEqual(arm_c.neff(np.eye(3)), 3.0)


if __name__ == "__main__":
    unittest.main()