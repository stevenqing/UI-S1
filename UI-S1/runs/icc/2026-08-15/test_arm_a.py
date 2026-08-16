import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("arm_a", RUN_DIR / "arm_a.py")
arm_a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arm_a
SPEC.loader.exec_module(arm_a)


class ArmATest(unittest.TestCase):
    def test_endpoint_classes(self):
        self.assertEqual(arm_a.endpoint_class(0.0, 0.0), "low_endpoint")
        self.assertEqual(arm_a.endpoint_class(1.0, 1.0), "high_endpoint")
        self.assertEqual(arm_a.endpoint_class(0.0, 1.0), "mixed_endpoint")
        self.assertEqual(arm_a.endpoint_class(0.4, 0.6), "interior")

    def test_neighbor_deltas(self):
        grid = [0.0, 1.0]
        scores = [
            {"rho_v": 0.0, "rho_l": 0.0, "accuracy": 0.5},
            {"rho_v": 0.0, "rho_l": 1.0, "accuracy": 0.4},
            {"rho_v": 1.0, "rho_l": 0.0, "accuracy": 0.6},
            {"rho_v": 1.0, "rho_l": 1.0, "accuracy": 0.3},
        ]
        value = arm_a.surface_record(0, scores, scores[0], grid)
        self.assertAlmostEqual(value["neighbor_accuracy_deltas"]["rho_v_higher"], 0.1)
        self.assertIsNone(value["neighbor_accuracy_deltas"]["rho_v_lower"])


if __name__ == "__main__":
    unittest.main()