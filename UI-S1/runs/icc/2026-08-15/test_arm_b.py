import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("arm_b", RUN_DIR / "arm_b.py")
arm_b = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arm_b
SPEC.loader.exec_module(arm_b)


class ArmBTest(unittest.TestCase):
    def test_outcome_classes(self):
        self.assertEqual(arm_b.outcome_class(True, False, True), "wrong_to_correct")
        self.assertEqual(arm_b.outcome_class(True, True, False), "correct_to_wrong")
        self.assertEqual(arm_b.outcome_class(False, True, True), "unchanged_correct")

    def test_direction_diversity(self):
        self.assertEqual(arm_b.direction_class([0, 3], [0, 1]), "diversity_increase")
        self.assertEqual(arm_b.direction_class([0, 1], [0, 3]), "diversity_decrease")

    def test_direction_concentration(self):
        self.assertEqual(arm_b.direction_class([0, 1, 3, 4], [0, 3, 6, 1]), "same_L_concentration_increase")

    def test_direction_substitution_and_same(self):
        self.assertEqual(arm_b.direction_class([0, 1], [1, 2]), "lineage_substitution")
        self.assertEqual(arm_b.direction_class([0, 3], [6, 9]), "composition_same")


if __name__ == "__main__":
    unittest.main()