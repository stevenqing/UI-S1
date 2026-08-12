import ast
import copy
import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch_recovery import build_command
from recovery_common import FORBIDDEN_IMPORT_PREFIXES, ROOT, RUN_DIR, assert_protected_process, load_config, validate_lane_rows


def reference(index):
    return {
        "stable_index": index,
        "id": f"row-{index}",
        "episode_id": f"episode-{index // 2}",
        "source_index": index + 10,
        "source_sha256": f"source-{index}",
        "image_sha256": f"image-{index}",
        "image_size": [1080, 2400],
    }


def artifact(index):
    value = reference(index)
    return {
        **value,
        "setting": "low",
        "model_id": "MODEL",
        "model_revision": "revision",
        "model_index_sha256": "index",
        "prediction": {
            "action": "click", "value": "", "position": [0.2, 0.3],
            "parse_ok": True, "pixel_position": [216, 720],
        },
        "text_prompt_sha256": "a" * 64,
        "model_prompt_sha256": "b" * 64,
        "num_shards": 1,
        "shard_index": 0,
    }


class RecoveryTest(unittest.TestCase):
    def test_frozen_config_and_exact_commands(self):
        config = load_config()
        self.assertEqual(assert_protected_process(config), config["protected_process"])
        for lane in config["lanes"].values():
            command = build_command(config, lane)
            self.assertEqual(command[0], str(ROOT / config["python"]))
            self.assertEqual(command[-1], "--resume")
            self.assertIn(str(ROOT / lane["destination"]), command)
            self.assertNotEqual((ROOT / lane["destination"]).resolve(), (ROOT / lane["seed_path"]).resolve())

    def test_recovery_code_has_no_scorer_or_evaluator_import(self):
        for name in (
            "recovery_common.py", "prepare_recovery.py", "launch_recovery.py", "finalize_recovery.py"
        ):
            tree = ast.parse((RUN_DIR / name).read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(any(value.startswith(FORBIDDEN_IMPORT_PREFIXES) for value in imports), (name, imports))

    def test_lane_validation_rejects_duplicate_and_provenance_drift(self):
        lane = {
            "model_id": "MODEL", "setting": "low", "model_revision": "revision",
            "model_index_sha256": "index", "seed_rows": 2,
        }
        rows = [artifact(0), artifact(1)]
        ordered = validate_lane_rows(rows, [reference(0), reference(1)], lane, require_complete=False)
        self.assertEqual([row["id"] for row in ordered], ["row-0", "row-1"])
        with self.assertRaises(ValueError):
            validate_lane_rows([rows[0], rows[0]], [reference(0), reference(1)], lane, require_complete=False)
        changed = copy.deepcopy(rows)
        changed[1]["image_sha256"] = "changed"
        with self.assertRaises(ValueError):
            validate_lane_rows(changed, [reference(0), reference(1)], lane, require_complete=False)
        changed = copy.deepcopy(rows)
        changed[1]["stable_index"] = 0
        with self.assertRaises(ValueError):
            validate_lane_rows(changed, [reference(0), reference(1)], lane, require_complete=False)

    def test_recovery_sources_do_not_access_ground_truth_fields(self):
        for name in (
            "recovery_common.py", "prepare_recovery.py", "launch_recovery.py", "finalize_recovery.py"
        ):
            tree = ast.parse((RUN_DIR / name).read_text())
            strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
            self.assertFalse(any(value.startswith("gt_") for value in strings), name)


if __name__ == "__main__":
    unittest.main()