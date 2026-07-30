import inspect
import unittest

from PIL import Image

from views import generate_view, max_visual_tokens


class ViewGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 200), (255, 255, 255))

    def test_signature_cannot_receive_gt(self):
        parameters = set(inspect.signature(generate_view).parameters)
        self.assertEqual(parameters, {"image", "view_id", "full_prediction_center"})
        source = inspect.getsource(generate_view)
        for forbidden in ("gt_action", "gt_bbox", "gt_input_text", "answer"):
            self.assertNotIn(forbidden, source)

    def test_border_inverse_mapping(self):
        view = generate_view(self.image, "v1")
        self.assertEqual(view.image.size, (156, 256))
        self.assertEqual(view.geometry.view_to_original_normalized(28, 28), (0.0, 0.0))
        self.assertEqual(view.geometry.view_to_original_normalized(128, 228), (1.0, 1.0))

    def test_tight_crop_inverse_mapping(self):
        view = generate_view(self.image, "v2", (50, 100))
        self.assertEqual(view.image.size, (50, 100))
        self.assertEqual(view.geometry.view_to_original_normalized(25, 50), (0.5, 0.5))

    def test_wide_crop_clamps_center_not_crop(self):
        view = generate_view(self.image, "v3", (-100, 1000))
        self.assertEqual(view.image.size, (75, 150))
        mapped = view.geometry.view_to_original_normalized(37.5, 75)
        self.assertLessEqual(abs(mapped[0] - 0.0), 0.5 / self.image.width)
        self.assertEqual(mapped[1], 1.0)

    def test_missing_center_falls_back_and_flags(self):
        view = generate_view(self.image, "v2", None)
        self.assertTrue(view.geometry.center_fallback)
        self.assertEqual(view.geometry.view_to_original_normalized(25, 50), (0.5, 0.5))

    def test_processor_profiles(self):
        self.assertEqual(max_visual_tokens("androidcontrol", "full"), 12800)
        self.assertEqual(max_visual_tokens("androidcontrol", "v4"), 768)
        self.assertEqual(max_visual_tokens("mind2web", "full"), 1344)
        self.assertEqual(max_visual_tokens("mind2web", "v4"), 768)


if __name__ == "__main__":
    unittest.main()