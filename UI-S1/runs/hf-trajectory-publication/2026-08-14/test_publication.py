import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_release", RUN_DIR / "prepare_release.py")
prepare_release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_release
SPEC.loader.exec_module(prepare_release)


class PublicationTest(unittest.TestCase):
    def test_frozen_selection_counts(self):
        selected = prepare_release.selected_artifacts()
        self.assertEqual(len(selected["mind2web"]), 44)
        self.assertEqual(len(selected["androidcontrol"]), 12)

    def test_forbidden_evaluation_fields(self):
        for key in ("target_bbox", "gt_point", "candidate_success", "reward", "private_label"):
            with self.assertRaises(ValueError):
                prepare_release.validate_value({key: 1})

    def test_model_output_fields_are_allowed(self):
        prepare_release.validate_value(
            {
                "image_sha256": "abc",
                "response": "CLICK <point>[[0.5, 0.5]]</point>",
                "prediction": {"action": "CLICK", "parse_ok": True, "position": [0.5, 0.5]},
            }
        )

    def test_local_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_release.validate_value({"image_path": "/scratch/private/image.png"})


if __name__ == "__main__":
    unittest.main()