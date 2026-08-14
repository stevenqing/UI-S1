import unittest

import numpy as np

from mask_common import (
    cohen_kappa, empty_mask, generalized_neff, informative_mask_pixels,
    nearest_pixels,
)


class MaskGeometryTest(unittest.TestCase):
    def test_information_circle_uses_pixel_centers(self):
        pixels = informative_mask_pixels(10, 10, (5.0, 5.0), 1.0)
        self.assertEqual(pixels.tolist(), [44, 45, 54, 55])

    def test_nearest_pixels_has_exact_area_and_stable_ties(self):
        pixels = nearest_pixels(10, 10, (5.0, 5.0), 4)
        self.assertEqual(pixels.tolist(), [44, 45, 54, 55])

    def test_empty_mask_matches_area_and_excludes_modes(self):
        result = empty_mask(100, 80, 25, (20.0, 20.0), [(20.0, 20.0), (30.0, 20.0)])
        self.assertIsNotNone(result)
        self.assertEqual(len(result["pixels"]), 25)
        mode_pixels = {20 * 100 + 20, 20 * 100 + 30}
        self.assertTrue(mode_pixels.isdisjoint(set(result["pixels"].tolist())))

    def test_generalized_neff_identity(self):
        self.assertEqual(generalized_neff(np.eye(3)), 3.0)

    def test_degenerate_kappa_is_undefined(self):
        self.assertIsNone(cohen_kappa([True, True], [True, True]))


if __name__ == "__main__":
    unittest.main()