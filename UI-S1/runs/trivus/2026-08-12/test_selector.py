import ast
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finalize_selector import validate_prediction
from selector_data import (
    assert_selector_environment, audit_public_record, build_prompt,
    deterministic_permutation, load_config, normalize_candidate,
    public_candidate_permutation, render_overlay, rendered_image_sha256,
)


class SelectorTest(unittest.TestCase):
    def record(self):
        return {
            "schema_version": 1,
            "sample_key": "androidcontrol/low/ac_1",
            "benchmark": "androidcontrol",
            "setting": "low",
            "row_id": "ac_1",
            "fold": 2,
            "group": "episode",
            "image_path": "unused.png",
            "image_sha256": "image",
            "instruction": "open settings",
            "history": "None",
            "candidates": [
                {"action": "click", "coordinate": [0.2, 0.3], "parameter": "", "parse_ok": True},
                {"action": "type", "coordinate": None, "parameter": "hello", "parse_ok": True},
                {"action": "press_back", "coordinate": None, "parameter": "", "parse_ok": True},
            ],
        }

    def test_config_and_permutation_are_frozen(self):
        config = load_config()
        self.assertEqual(assert_selector_environment(config), sys.executable)
        first = deterministic_permutation("row", config["seed"])
        self.assertEqual(first, deterministic_permutation("row", config["seed"]))
        self.assertEqual(sorted(first), [0, 1, 2])
        self.assertEqual(sorted(public_candidate_permutation("row", config["seed"])), [0, 1, 2])

    def test_public_schema_rejects_identity_and_target_fields(self):
        record = self.record()
        self.assertTrue(audit_public_record(record, verify_image=False))
        for field in ("source", "model_id", "slot", "fallback", "target_bbox", "gt_action"):
            changed = copy.deepcopy(record)
            changed[field] = "forbidden"
            with self.assertRaises(ValueError):
                audit_public_record(changed, verify_image=False)

    def test_candidate_normalization_is_action_semantic(self):
        config = load_config()
        click = normalize_candidate({"action": "click", "position": [0.2, 0.3], "value": "[2,3]", "parse_ok": True}, config)
        self.assertEqual(click["coordinate"], [0.2, 0.3])
        self.assertEqual(click["parameter"], "")
        typed = normalize_candidate({"action": "type", "position": [0.2, 0.3], "value": "hello", "parse_ok": True}, config)
        self.assertIsNone(typed["coordinate"])
        self.assertEqual(typed["parameter"], "hello")
        with self.assertRaises(ValueError):
            normalize_candidate({"action": "click", "position": [1.2, 0.3], "value": "", "parse_ok": True}, config)

    def test_prompt_contains_no_identity_or_fallback(self):
        prompt = build_prompt(self.record(), (2, 0, 1))
        lowered = prompt.lower()
        for token in ("ui-agile", "gui-r1", "ui-r1", "source", "model", "slot", "fallback", "ground truth"):
            self.assertNotIn(token, lowered)

    def test_rendered_overlay_hash_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "screen.png"
            Image.new("RGB", (200, 400), "white").save(image_path)
            record = self.record()
            record["image_path"] = str(image_path)
            first = rendered_image_sha256(render_overlay(record, (2, 0, 1), 1600))
            second = rendered_image_sha256(render_overlay(record, (2, 0, 1), 1600))
            self.assertEqual(first, second)
            self.assertNotEqual(first, rendered_image_sha256(render_overlay(record, (0, 2, 1), 1600)))

    def test_prediction_validation(self):
        config = load_config()
        public = self.record()
        prediction = {
            "schema_version": 1,
            "sample_key": public["sample_key"],
            "benchmark": public["benchmark"],
            "setting": public["setting"],
            "row_id": public["row_id"],
            "fold": public["fold"],
            "group": public["group"],
            "display_to_candidate": [2, 0, 1],
            "selected_label": "B",
            "selected_candidate_index": 0,
            "label_logits": [1.0, 2.0, 0.0],
            "label_probabilities": [0.25, 0.6, 0.15],
            "prompt_sha256": hashlib.sha256(build_prompt(public, [2, 0, 1]).encode()).hexdigest(),
            "overlay_sha256": "b" * 64,
            "image_sha256": public["image_sha256"],
            "model_index_sha256": config["model"]["index_sha256"],
        }
        validate_prediction(prediction, public, config, verify_overlay=False)
        changed = copy.deepcopy(prediction)
        changed["label_probabilities"] = [0.2, 0.2, 0.2]
        with self.assertRaises(ValueError):
            validate_prediction(changed, public, config, verify_overlay=False)
        changed = copy.deepcopy(prediction)
        changed["selected_label"] = "A"
        with self.assertRaises(ValueError):
            validate_prediction(changed, public, config, verify_overlay=False)
        changed = copy.deepcopy(prediction)
        changed["prompt_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_prediction(changed, public, config, verify_overlay=False)

    def test_prelock_sources_do_not_import_scorers_or_index_gt(self):
        run_dir = Path(__file__).resolve().parent
        for name in ("selector_data.py", "prepare_selector_public.py", "selector_infer.py", "finalize_selector.py"):
            tree = ast.parse((run_dir / name).read_text())
            imports = []
            subscripts = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    subscripts.append(node.slice.value)
            self.assertFalse(any("scoring" in value or "evaluator" in value for value in imports), (name, imports))
            self.assertFalse(any(value.startswith("gt_") for value in subscripts), (name, subscripts))


if __name__ == "__main__":
    unittest.main()