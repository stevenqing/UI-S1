import unittest
from types import SimpleNamespace

from gui360_long_horizon.overnight import _action_only_history_line, _corrupt_history, _history_line, _native_action_history_line, _pred_dict_from_text, latest_by_plan_key, latest_by_textdrift_key, latest_by_textmem_key, summarize_plan_rows, summarize_textdrift_rows, summarize_textmem_rows


class OvernightParserTests(unittest.TestCase):
    def test_parse_fenced_action_json(self):
        pred = _pred_dict_from_text('```json\n{"action":"click","coordinate":[1,2]}\n```')
        self.assertEqual(pred["function"], "click")
        self.assertEqual(pred["coordinate"], [1, 2])

    def test_parse_sft_function_then_args_json(self):
        text = '<tool_call>\nclick\n<tool_call>\n{"coordinate":[848,136]}\n</tool_call>'
        pred = _pred_dict_from_text(text)
        self.assertEqual(pred["function"], "click")
        self.assertEqual(pred["coordinate"], [848, 136])

    def test_parse_sft_unclosed_args_json(self):
        text = '<tool_call>\nclick\n<tool_call>\n{\n  "coordinate": [337, 205]\n}\n<tool_call>\ntype\n<tool_call>\n{"text":"Text"}'
        pred = _pred_dict_from_text(text)
        self.assertEqual(pred["function"], "click")
        self.assertEqual(pred["coordinate"], [337, 205])

    def test_history_line_and_corrupt_history(self):
        step = SimpleNamespace(step_id=2, subtask="Click OK", observation="screen", gt_action={"function": "click"}, gt_xy=(10.0, 20.0))
        line = _history_line(step)
        self.assertIn("Click OK", line)
        self.assertIn("coordinate=[10, 20]", line)
        self.assertIn("injected_error=true", _corrupt_history(line, step))

    def test_native_history_line(self):
        step = SimpleNamespace(step_id=3, gt_function="click", gt_action={"function": "click"}, gt_xy=(12.0, 34.0))
        self.assertEqual(_native_action_history_line(step), "Step 3: click(coordinate=[12, 34])")
        self.assertEqual(_action_only_history_line(step), "3: click [12, 34]")

    def test_textmem_summary_gate(self):
        rows = [
            {"step_uid": "a", "ok": True, "correct": True, "cond": {"history_mode": "none"}},
            {"step_uid": "b", "ok": True, "correct": False, "cond": {"history_mode": "none"}},
            {"step_uid": "a", "ok": True, "correct": True, "cond": {"history_mode": "full"}},
            {"step_uid": "b", "ok": True, "correct": False, "cond": {"history_mode": "full"}},
        ]
        summary = summarize_textmem_rows(rows, 0.01)
        self.assertTrue(summary["gate_passed"])
        self.assertEqual(summary["full_minus_none"], 0.0)

    def test_textmem_latest_keeps_modes(self):
        rows = [
            {"step_uid": "a", "ok": True, "correct": False, "cond": {"history_mode": "none"}},
            {"step_uid": "a", "ok": True, "correct": True, "cond": {"history_mode": "full"}},
            {"step_uid": "a", "ok": True, "correct": True, "cond": {"history_mode": "none"}},
        ]
        latest = latest_by_textmem_key(rows)
        self.assertEqual(len(latest), 2)
        by_mode = {row["cond"]["history_mode"]: row for row in latest}
        self.assertTrue(by_mode["none"]["correct"])
        self.assertTrue(by_mode["full"]["correct"])

    def test_textdrift_summary(self):
        rows = [
            {"step_uid": "a", "ok": True, "correct": True, "cond": {"injected_error": 1}},
            {"step_uid": "a", "ok": True, "correct": False, "cond": {"injected_error": 2}},
            {"step_uid": "a", "ok": True, "correct": True, "cond": {"injected_error": 1}},
        ]
        latest = latest_by_textdrift_key(rows)
        self.assertEqual(len(latest), 2)
        summary = summarize_textdrift_rows(latest, n_base_steps=1)
        self.assertEqual(summary["acc_by_injected"]["1"], 1.0)
        self.assertEqual(summary["acc_by_injected"]["2"], 0.0)

    def test_plan_summary(self):
        rows = [
            {"step_uid": "a", "ok": True, "correct": False, "cond": {"plan": "none"}},
            {"step_uid": "a", "ok": True, "correct": True, "cond": {"plan": "oracle"}},
            {"step_uid": "a", "ok": True, "correct": True, "cond": {"plan": "none"}},
        ]
        latest = latest_by_plan_key(rows)
        self.assertEqual(len(latest), 2)
        summary = summarize_plan_rows(latest)
        self.assertEqual(summary["acc_by_plan"]["none"], 1.0)
        self.assertEqual(summary["acc_by_plan"]["oracle"], 1.0)
        self.assertEqual(summary["oracle_minus_none"], 0.0)


if __name__ == "__main__":
    unittest.main()
