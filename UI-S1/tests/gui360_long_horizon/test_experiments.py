import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gui360_long_horizon.experiments import m_core, m_diag, m_plan, m_recover, m_textdrift, m_textmem
from gui360_long_horizon.harness.model import Decode
from gui360_long_horizon.types import Row


class DummyModel:
    cache_id = "dummy-experiments"
    model_name = "dummy-model"

    def generate(self, messages, n=1, logprobs=False, **kwargs):
        text = '<tool_call>{"function":"click","args":{"coordinate":[20,30]},"status":"CONTINUE"}</tool_call>'
        return [Decode(text=text) for _ in range(n)]


def _step(exec_id="e1", step_id=1, gt=True, off=False, raw=None):
    controls = [{"control_type": "Button", "control_text": "OK", "control_rect": [10, 20, 30, 40]}]
    if off:
        controls = [
            {"control_type": "Dialog", "control_text": "Error: invalid value"},
            {"control_type": "Button", "control_text": "OK", "control_rect": [10, 20, 30, 40]},
        ]
    return SimpleNamespace(
        exec_id=exec_id,
        app="excel",
        tag="in_app",
        step_id=step_id,
        total_steps=3,
        request="Do task",
        template="template.xlsx",
        subtask="Click OK",
        observation="screen",
        thought="click ok",
        image_rel_path="test/image/x.png",
        control_infos={"uia_controls_info": controls},
        gt_function="click" if gt else None,
        gt_rect=(10, 20, 30, 40) if gt else None,
        raw={"_image_data_url": "data:image/png;base64,AAA=", **(raw or {})},
    )


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()

    def test_row_contract_rejects_off_step_correctness(self):
        with self.assertRaises(ValueError):
            Row("e", "excel", "in_app", "task", 1, 0.1, "off", 0, 0, step_correct=True)

    @patch("gui360_long_horizon.harness.predict._offtrack_probability", return_value=0.0)
    def test_textmem_gate_emits_all_history_modes(self, _probe):
        result = m_textmem.run([_step()], self.model, bucket_fn=lambda step: 1, n=2)
        self.assertEqual(len(result.rows), 4)
        self.assertTrue(result.gate_passed)
        self.assertEqual(set(result.acc_by_mode), {"full", "summary", "corrupt", "none"})

    @patch("gui360_long_horizon.harness.predict._offtrack_probability", return_value=0.0)
    def test_plan_core_and_textdrift_emit_on_rows(self, _probe):
        traj = [_step(step_id=1), _step(step_id=2)]
        self.assertEqual(len(m_plan.run([traj], self.model, n=1).rows), 4)
        core = m_core.run([traj], self.model, n=1)
        self.assertIn("plan", core.blocks)
        self.assertTrue(all(row.manifold == "on" for row in core.rows))
        drift = m_textdrift.run([traj[0]], [_step("f", gt=False, off=True)], self.model, max_injected=2, n=1)
        self.assertEqual([row.cond["injected_error"] for row in drift.rows], [1, 2])

    @patch("gui360_long_horizon.harness.predict._offtrack_probability", return_value=0.8)
    def test_recover_and_diag_keep_off_step_correct_none(self, _probe):
        off = _step("f1", step_id=2, gt=False, off=True)
        recover = m_recover.run([off], self.model, t_star_by_exec={"f1": 2}, n=1)
        self.assertEqual(len(recover.rows), 1)
        self.assertIsNone(recover.rows[0].step_correct)
        self.assertTrue(recover.rows[0].recovery_correct)

        on = _step("s1", step_id=2)
        diag = m_diag.run([on], [off], self.model, t_star_by_exec={"f1": 2}, bucket_fn=lambda step: 0, n=1)
        self.assertFalse(diag.identifies_drift)
        self.assertTrue(all(row.step_correct is None for row in diag.rows))
        with self.assertRaises(RuntimeError):
            _ = diag.drift_effect


if __name__ == "__main__":
    unittest.main()
