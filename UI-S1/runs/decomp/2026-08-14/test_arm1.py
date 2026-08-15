import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("arm1", RUN_DIR / "arm1.py")
arm1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arm1
SPEC.loader.exec_module(arm1)


class Arm1Test(unittest.TestCase):
    def test_subset_features(self):
        value = arm1.subset_features((1 << 0) | (1 << 1) | (1 << 3))
        self.assertEqual(value["budget"], 3)
        self.assertEqual(value["lineage_count"], 2)
        self.assertEqual(value["view_count"], 2)

    def test_choose_cell(self):
        scores = {(2, 1, 2): 0.5, (2, 2, 1): 0.5}
        variance = {(2, 1, 2): 1.0, (2, 2, 1): 0.5}
        self.assertEqual(arm1.choose_cell(scores, variance), (2, 2, 1))

    def test_anova_constant_is_na(self):
        features = [arm1.subset_features(mask) for mask in (3, 5, 6)]
        self.assertIsNone(arm1.anova_components([1, 1, 1], features)["lineage"])

    def test_marginal_contrast(self):
        scores = {(3, 1, 2): 0.4, (3, 2, 2): 0.5, (3, 2, 3): 0.55}
        value = arm1.marginal_contrasts(scores)
        self.assertAlmostEqual(value["lineage"], 0.1)
        self.assertAlmostEqual(value["view"], 0.05)

    def test_application_multiplicities_preserve_fold_draw_counts(self):
        applications = ["a", "b", "c", "d"]
        folds = {"a": 0, "b": 0, "c": 1, "d": 1}
        values = arm1.application_multiplicities(applications, folds, 3, 7)
        self.assertTrue((values[:, :2].sum(axis=1) == 2).all())
        self.assertTrue((values[:, 2:].sum(axis=1) == 2).all())

    def test_application_multiplicities_allow_empty_fold(self):
        values = arm1.application_multiplicities(["a"], {"a": 0}, 2, 7)
        self.assertTrue((values == 1).all())


if __name__ == "__main__":
    unittest.main()