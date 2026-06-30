import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import tarfile

from PIL import Image

from gui360_long_horizon.data.loader import _make_step, _norm_path, list_shard_files, load_image


def _raw_row(split="test", complete="yes", action=True):
    step = {
        "screenshot_clean": "success\\excel_1/action_step1.png",
        "screenshot_desktop": "success\\excel_1/desktop_action_step1.png",
        "screenshot_annotated": "success\\excel_1/action_step1_annotated.png",
        "screenshot_selected_controls": "success\\excel_1/action_step1_selected.png",
        "ui_tree": {"name": "root"},
        "control_infos": {"uia_controls_info": [{"label": 1, "control_type": "Button", "control_text": "OK", "control_rect": [1, 2, 3, 4]}]},
        "subtask": "click OK",
        "observation": "screen",
        "thought": "need ok",
        "status": "CONTINUE",
        "tags": [],
    }
    if action:
        step["action"] = {
            "function": "click",
            "coordinate_x": 10.5,
            "coordinate_y": 20.5,
            "rectangle": {"left": 1, "top": 2, "right": 30, "bottom": 40},
        }
    return {
        "execution_id": "excel_1",
        "app_domain": "excel",
        "request": "Click OK.",
        "template": "template.xlsx",
        "step_id": 1,
        "total_steps": 1,
        "evaluation": {"complete": complete},
        "step": step,
    }


class LoaderTests(unittest.TestCase):
    def test_success_step_has_gt_fields_and_normalized_image_paths(self):
        step = _make_step(_raw_row(), "test", "excel", "in_app", contiguous=True)
        self.assertTrue(step.traj_success)
        self.assertEqual(step.gt_function, "click")
        self.assertEqual(step.gt_xy, (10.5, 20.5))
        self.assertEqual(step.gt_rect, (1.0, 2.0, 30.0, 40.0))
        self.assertEqual(step.screenshot_clean, "success/excel_1/action_step1.png")
        self.assertEqual(step.image_rel_path, "test/image/excel/in_app/success/excel_1/action_step1.png")
        self.assertTrue(step.has_a11y)
        self.assertTrue(step.contiguous)


    def test_fail_step_has_no_gt_even_if_raw_action_present(self):
        step = _make_step(_raw_row(split="fail", complete="no", action=True), "fail", "excel", "in_app", contiguous=True)
        self.assertFalse(step.traj_success)
        self.assertIsNone(step.gt_action)
        self.assertIsNone(step.gt_function)
        self.assertIsNone(step.gt_xy)
        self.assertIsNone(step.gt_rect)


    def test_norm_path_handles_windows_separators(self):
        self.assertEqual(_norm_path("images\\excel_1/action_step4.png"), "images/excel_1/action_step4.png")

    def test_list_shard_files_prefers_local_cache(self):
        with TemporaryDirectory() as tmpdir:
            shard = Path(tmpdir) / "fail/data/excel/in_app/fail/excel_1.jsonl"
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_text("{}\n", encoding="utf-8")
            with patch("gui360_long_horizon.data.loader.DEFAULT_CACHE_DIR", tmpdir), patch("gui360_long_horizon.data.loader.HfApi", side_effect=AssertionError("should not use HF")):
                self.assertEqual(list_shard_files("dummy/repo", "fail", "excel", "in_app"), ["fail/data/excel/in_app/fail/excel_1.jsonl"])

    def test_list_shard_files_can_force_remote_listing(self):
        class FakeApi:
            def list_repo_tree(self, repo_id, repo_type, path_in_repo, recursive):
                return ["fail/data/excel/in_app/fail/remote.jsonl"]

        with TemporaryDirectory() as tmpdir:
            shard = Path(tmpdir) / "fail/data/excel/in_app/fail/local.jsonl"
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_text("{}\n", encoding="utf-8")
            with patch("gui360_long_horizon.data.loader.DEFAULT_CACHE_DIR", tmpdir), patch("gui360_long_horizon.data.loader.HfApi", return_value=FakeApi()), patch.dict("os.environ", {"GUI360_FORCE_REMOTE_SHARDS": "1"}):
                self.assertEqual(list_shard_files("dummy/repo", "fail", "excel", "in_app"), ["fail/data/excel/in_app/fail/remote.jsonl"])

    def test_load_image_opens_normalized_direct_file(self):
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (3, 2), color=(255, 0, 0)).save(image_path)

            def fake_download(repo, filename, cache_dir):
                self.assertEqual(filename, "test/image/excel/in_app/success/excel_1/action_step1.png")
                return image_path

            with patch("gui360_long_horizon.data.loader._download", side_effect=fake_download):
                image = load_image("dummy/repo", "test\\image\\excel\\in_app\\success\\excel_1\\action_step1.png")
            self.assertEqual(image.size, (3, 2))

    def test_load_image_prefers_local_cache_file(self):
        with TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "cache"
            image_path = cache / "fail/image/excel/in_app/fail/excel_1/action_step1.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (6, 7), color=(0, 0, 255)).save(image_path)
            with patch("gui360_long_horizon.data.loader.DEFAULT_CACHE_DIR", str(cache)), patch("gui360_long_horizon.data.loader._download", side_effect=AssertionError("should not download")):
                image = load_image("dummy/repo", "fail/image/excel/in_app/fail/excel_1/action_step1.png")
            self.assertEqual(image.size, (6, 7))

    def test_load_image_assembles_split_tar(self):
        with TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "cache"
            source = Path(tmpdir) / "source.tar.gz"
            png = Path(tmpdir) / "action_step1.png"
            Image.new("RGB", (4, 5), color=(0, 255, 0)).save(png)
            with tarfile.open(source, "w:gz") as tar:
                tar.add(png, arcname="image/excel/in_app/fail/excel_1/action_step1.png")
            blob = source.read_bytes()
            midpoint = len(blob) // 2
            part0 = Path(tmpdir) / "part0"
            part1 = Path(tmpdir) / "part1"
            part0.write_bytes(blob[:midpoint])
            part1.write_bytes(blob[midpoint:])

            def fake_download(repo, filename, cache_dir):
                if filename == "fail/image/excel/in_app/fail/excel_1/action_step1.png":
                    raise FileNotFoundError(filename)
                if filename == "fail/image.tar.gz":
                    raise FileNotFoundError(filename)
                if filename == "fail/image.tar.gz000":
                    target = cache / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(part0.read_bytes())
                    return target
                if filename == "fail/image.tar.gz001":
                    target = cache / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(part1.read_bytes())
                    return target
                raise FileNotFoundError(filename)

            with patch("gui360_long_horizon.data.loader.DEFAULT_CACHE_DIR", str(cache)), patch("gui360_long_horizon.data.loader._download", side_effect=fake_download):
                image = load_image("dummy/repo", "fail/image/excel/in_app/fail/excel_1/action_step1.png")
            self.assertEqual(image.size, (4, 5))


if __name__ == "__main__":
    unittest.main()
