import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from gui360_long_horizon.stages import loader_smoke


def _step(split="test", gt=True):
    return SimpleNamespace(
        exec_id=f"{split}-exec",
        step_id=1,
        image_rel_path=f"{split}/image/excel/in_app/success/x/action_step1.png",
        contiguous=True,
        gt_xy=(1, 2) if gt else None,
        gt_rect=(0, 0, 3, 4) if gt else None,
    )


class StageRunnerTests(unittest.TestCase):
    def test_loader_smoke_passes_test_and_fail_invariants(self):
        def fake_load(repo, split, app, tag, limit=None):
            return {f"{split}-exec": [_step(split, gt=split != "fail")]}

        with patch("gui360_long_horizon.stages.load_trajectories", side_effect=fake_load), patch("gui360_long_horizon.stages.load_image", return_value=Image.new("RGB", (2, 2))):
            result = loader_smoke({"repo": "dummy", "shards": [{"app": "excel", "tag": "in_app", "splits": ["test", "fail"]}]})
        self.assertTrue(result.passed)
        self.assertEqual(len(result.details["splits"]), 2)

    def test_loader_smoke_fails_missing_image(self):
        def fake_load(repo, split, app, tag, limit=None):
            return {f"{split}-exec": [_step(split, gt=split != "fail")]}

        with patch("gui360_long_horizon.stages.load_trajectories", side_effect=fake_load), patch("gui360_long_horizon.stages.load_image", side_effect=FileNotFoundError("missing")):
            result = loader_smoke({"repo": "dummy", "shards": [{"app": "excel", "tag": "in_app", "splits": ["test", "fail"]}]})
        self.assertFalse(result.passed)
        self.assertFalse(result.details["splits"][0]["image_checks"][0]["ok"])


if __name__ == "__main__":
    unittest.main()
