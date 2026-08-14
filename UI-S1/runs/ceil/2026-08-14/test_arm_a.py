import unittest

import numpy as np

from arm_a import cohen_kappa_from_counts, curve_summary, parametric_fit


class ArmANumericsTest(unittest.TestCase):
    def test_kappa_counts(self):
        value, missing = cohen_kappa_from_counts(4, 2, 2, 2)
        self.assertFalse(missing)
        self.assertEqual(value, 1.0)

    def test_parametric_fit_recovers_saturating_curve(self):
        x_values = np.linspace(1, 3, 20)
        y_values = 0.8 - 0.4 * np.exp(-0.7 * x_values)
        fit = parametric_fit(x_values, y_values)
        self.assertIsNotNone(fit)
        _, a_value, b_value, c_value = fit
        self.assertAlmostEqual(a_value, 0.8, places=5)
        self.assertAlmostEqual(b_value, 0.4, places=5)
        self.assertAlmostEqual(c_value, 0.7, places=5)

    def test_curve_summary_has_finite_delta(self):
        x_values = np.linspace(1, 2, 4095)
        y_values = 0.7 - 0.2 * np.exp(-x_values)
        report = curve_summary(x_values, y_values)
        self.assertIsNotNone(report["Delta_infinity"])
        self.assertGreaterEqual(report["isotonic"]["boundary_slope"], 0)


if __name__ == "__main__":
    unittest.main()