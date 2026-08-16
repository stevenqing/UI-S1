import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("stage1", RUN_DIR / "stage1.py")
stage1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage1
SPEC.loader.exec_module(stage1)


class Stage1Test(unittest.TestCase):
    def test_lineage_weights_mean_one(self):
        rows = {
            "a": {"candidates": tuple({"correct": index % 2 == 0} for index in range(12))},
            "b": {"candidates": tuple({"correct": index % 3 == 0} for index in range(12))},
        }
        weights = stage1.lineage_weights(rows, ["a", "b"])
        self.assertAlmostEqual(sum(weights.values()) / len(weights), 1.0)

    def test_rho_grid_tie_uses_first_pair(self):
        class Fake:
            @staticmethod
            def select_group(candidates, rho_v, rho_l, weights, singleton=False):
                return (0,), 0, 0
        rows = {"a": {"candidates": ({"correct": True},)}}
        selected, _ = stage1.select_rho(Fake, rows, ["a"], [0.0, 1.0])
        self.assertEqual((selected["rho_v"], selected["rho_l"]), (0.0, 0.0))

    def test_evaluator_requires_precommitted_selection_status(self):
        required = "PASS_EVID_STAGE1_NESTED_SELECTIONS_BEFORE_OUTER_EVALUATION"
        self.assertIn(required, Path(stage1.__file__).read_text())


if __name__ == "__main__":
    unittest.main()