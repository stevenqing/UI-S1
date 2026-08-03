import math
import unittest

import numpy as np

from neff import cohen_kappa, effective_sample_size, linear_fit


class NeffContractTest(unittest.TestCase):
    def test_equicorrelation_formula(self):
        self.assertAlmostEqual(effective_sample_size(12, 0.895), 12 / (1 + 11 * 0.895))
        self.assertAlmostEqual(effective_sample_size(8, 0.0), 8.0)

    def test_kappa_identical_and_opposed(self):
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [1, 1, 0, 0]), -1.0)

    def test_linear_fit_exact(self):
        points = [{"x": value, "accuracy": 2 + 3 * value} for value in range(5)]
        fit = linear_fit(points, "x")
        self.assertAlmostEqual(fit["intercept"], 2.0)
        self.assertAlmostEqual(fit["coefficient"], 3.0)
        self.assertAlmostEqual(fit["residual_sd"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
