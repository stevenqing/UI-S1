import json
import tempfile
import unittest
from pathlib import Path

from gui360_long_horizon.analysis.controls import DEFAULT_DEPENDENCY_THRESHOLDS, ThresholdFreezeError, assert_thresholds_frozen, thresholds_to_dict
from gui360_long_horizon.analysis.dependency_diag import load_episode_json, run_dependency_diagnostic, write_dependency_verdict
from gui360_long_horizon.data.availability import OcrReferee
from gui360_long_horizon.data.defect_localize import prediction_records_for_steps


def _step(exec_id, step_id, action, controls=None, request="Do private workflow", template="tmpl", function_role="Edit"):
    control_list = controls if controls is not None else [{"control_type": function_role, "control_text": "private field", "control_rect": [0, 0, 10, 10]}]
    return {
        "exec_id": exec_id,
        "step_id": step_id,
        "request": request,
        "template": template,
        "gt_action": action,
        "gt_function": action.get("function"),
        "gt_xy": action.get("coordinate", [1, 1]),
        "control_infos": {"uia_controls_info": control_list},
    }


class DependencyDiagnosticTests(unittest.TestCase):
    def test_exact_bucket_accounting_and_no_battlefield(self):
        episode = [
            _step("e1", 0, {"function": "type", "args": {"text": "alpha42"}, "coordinate": [1, 1]}),
            _step("e1", 1, {"function": "type", "args": {"text": "alpha42"}, "coordinate": [1, 1]}, controls=[
                {"control_type": "Edit", "control_text": "private field", "control_rect": [0, 0, 10, 10]},
                {"control_type": "Text", "control_text": "alpha42", "control_rect": [20, 20, 30, 30]},
            ]),
        ]
        payload = run_dependency_diagnostic([episode], ocr=OcrReferee(cache={}))
        self.assertEqual(payload["bucket_total"], payload["q1"]["candidate_total"])
        self.assertEqual(payload["bucket_counts"]["onscreen_a11y"], 1)
        self.assertEqual(payload["q1"]["survivor_n"], 0)
        self.assertEqual(payload["verdict"]["label"], "NO_BATTLEFIELD")

    def test_survivor_then_memory_defect_stays_marginal_when_small(self):
        episode = [
            _step("e2", 0, {"function": "type", "args": {"text": "secret42"}, "coordinate": [1, 1]}),
            _step("e2", 1, {"function": "swipe", "coordinate": [1, 1]}, controls=[]),
            _step("e2", 2, {"function": "click", "coordinate": [2, 2]}, controls=[]),
            _step("e2", 3, {"function": "type", "args": {"text": "secret42"}, "coordinate": [1, 1]}, controls=[
                {"control_type": "Edit", "control_text": "private field", "control_rect": [0, 0, 10, 10]},
                {"control_type": "Edit", "control_text": "secondary field", "control_rect": [20, 20, 30, 30]},
            ]),
        ]
        correct = {"e2:step0": True, "e2:step1": True, "e2:step2": True, "e2:step3": False}
        pred = {"e2:step3": {"function": "type", "text": "wrong42"}}
        records = prediction_records_for_steps(load_episode_json_from_steps(episode), correct, pred)
        ocr = OcrReferee(cache={"e2:step1": "", "e2:step2": "", "e2:step3": "destination"})
        payload = run_dependency_diagnostic([load_episode_json_from_steps(episode)], ocr=ocr, prediction_rows=records)
        self.assertEqual(payload["bucket_counts"]["survivor"], 1)
        self.assertEqual(payload["q3"]["bucket_counts"]["memory"], 1)
        self.assertEqual(payload["verdict"]["label"], "MARGINAL")

    def test_missing_controls_do_not_imply_forced_action_space(self):
        episode = [
            _step("e3", 0, {"function": "type", "args": {"text": "omega77"}, "coordinate": [1, 1]}, controls=[]),
            _step("e3", 1, {"function": "click", "coordinate": [2, 2]}, controls=[]),
            _step("e3", 2, {"function": "click", "coordinate": [3, 3]}, controls=[]),
            _step("e3", 3, {"function": "type", "args": {"text": "omega77"}, "coordinate": [1, 1]}, controls=[]),
        ]
        ocr = OcrReferee(cache={"e3:step0": "", "e3:step1": "", "e3:step2": "", "e3:step3": ""})
        payload = run_dependency_diagnostic([episode], ocr=ocr)
        self.assertEqual(payload["bucket_counts"]["forced"], 0)
        self.assertEqual(payload["bucket_counts"]["survivor"], 1)

    def test_keyboard_commands_are_not_carried_values(self):
        episode = [
            _step("e4", 0, {"function": "type", "args": {"text": "{ENTER}"}, "coordinate": [1, 1]}, controls=[]),
            _step("e4", 1, {"function": "click", "coordinate": [2, 2]}, controls=[]),
            _step("e4", 2, {"function": "click", "coordinate": [3, 3]}, controls=[]),
            _step("e4", 3, {"function": "type", "args": {"text": "{ENTER}"}, "coordinate": [1, 1]}, controls=[]),
        ]
        payload = run_dependency_diagnostic([episode], ocr=OcrReferee(cache={"e4:step0": "", "e4:step1": "", "e4:step2": "", "e4:step3": ""}))
        self.assertEqual(payload["bucket_counts"]["given"], 1)
        self.assertEqual(payload["bucket_counts"]["survivor"], 0)

    def test_keyboard_prefix_is_stripped_from_semantic_text(self):
        episode = [
            _step("e5", 0, {"function": "type", "args": {"text": "^a{BACKSPACE}omega77"}, "coordinate": [1, 1]}, controls=[]),
            _step("e5", 1, {"function": "click", "coordinate": [2, 2]}, controls=[]),
            _step("e5", 2, {"function": "click", "coordinate": [3, 3]}, controls=[]),
            _step("e5", 3, {"function": "type", "args": {"text": "omega77"}, "coordinate": [1, 1]}, controls=[]),
        ]
        payload = run_dependency_diagnostic([episode], ocr=OcrReferee(cache={"e5:step0": "", "e5:step1": "", "e5:step2": "", "e5:step3": ""}))
        self.assertEqual(payload["bucket_counts"]["given"], 0)
        self.assertEqual(payload["bucket_counts"]["survivor"], 1)

    def test_threshold_freeze_rejects_override_and_writer(self):
        data = thresholds_to_dict(DEFAULT_DEPENDENCY_THRESHOLDS)
        self.assertEqual(assert_thresholds_frozen(data), DEFAULT_DEPENDENCY_THRESHOLDS)
        data["q1_battlefield_share_min"] = 0.01
        with self.assertRaises(ThresholdFreezeError):
            assert_thresholds_frozen(data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dependency_verdict.json"
            write_dependency_verdict({"verdict": {"label": "NO_BATTLEFIELD"}}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["verdict"]["label"], "NO_BATTLEFIELD")


def load_episode_json_from_steps(steps):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "episodes.json"
        path.write_text(json.dumps([{"episode_id": "e2", "steps": steps}]), encoding="utf-8")
        return load_episode_json(path)[0]


if __name__ == "__main__":
    unittest.main()