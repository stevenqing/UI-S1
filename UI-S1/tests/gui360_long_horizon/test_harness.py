import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gui360_long_horizon.harness.correctness import coordinate_hit, function_match, step_correct
from gui360_long_horizon.harness.model import Decode
from gui360_long_horizon.harness.predict import query_step
from gui360_long_horizon.harness.prompt import VisualOnlyPromptError, build_messages


class DummyModel:
    cache_id = "dummy"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, n=1, logprobs=False, **kwargs):
        self.calls += 1
        text = '<tool_call>{"function":"click","args":{"coordinate":[10,20]},"status":"CONTINUE"}</tool_call>'
        return [Decode(text=text) for _ in range(n)]


def _step(gt=True):
    raw = {"_image_data_url": "data:image/png;base64,AAA=", "history_text": "Step 1: click()"}
    return SimpleNamespace(
        exec_id="e1",
        step_id=1,
        request="Click OK",
        subtask="Click OK",
        gt_function="click" if gt else None,
        gt_rect=(0, 0, 30, 30) if gt else None,
        image_rel_path="test/image/x.png",
        control_infos={"uia_controls_info": [{"label": 1, "control_type": "Button", "control_text": "OK", "control_rect": [0, 0, 30, 30]}]},
        raw=raw,
    )


class HarnessTests(unittest.TestCase):
    def test_step_correct_raises_on_gt_absent(self):
        with self.assertRaises(ValueError):
            step_correct({"function": "click", "coordinate": [1, 2]}, _step(gt=False))

    def test_function_and_coordinate_match(self):
        self.assertTrue(function_match({"function": "double_click"}, "click"))
        self.assertTrue(coordinate_hit((10, 20), (0, 0, 30, 30)))
        self.assertTrue(step_correct({"function": "click", "coordinate": [10, 20]}, _step()))

    def test_type_step_requires_text_match(self):
        step = _step()
        step.gt_function = "type"
        step.gt_action = {"function": "type", "args": {"text": "alpha42"}}
        self.assertFalse(step_correct({"function": "type", "coordinate": [10, 20], "args": {"text": "wrong"}}, step))
        self.assertTrue(step_correct({"function": "type", "coordinate": [10, 20], "args": {"text": "alpha42"}}, step))

    def test_prompt_a11y_mode_raises(self):
        visual = build_messages(_step(), "full", "visual")
        self.assertNotIn("Accessibility elements", visual[0]["content"][1]["text"])
        with self.assertRaises(VisualOnlyPromptError):
            build_messages(_step(), "full", "visual_a11y")

    def test_query_step_cache_reuses_identical_inputs(self):
        model = DummyModel()
        with patch("gui360_long_horizon.harness.predict._offtrack_probability", return_value=0.0):
            pred1 = query_step(model, _step(), "full", "visual", n=2)
            pred2 = query_step(model, _step(), "full", "visual", n=2)
        self.assertEqual(pred1.xy, (10.0, 20.0))
        self.assertEqual(pred1, pred2)
        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
