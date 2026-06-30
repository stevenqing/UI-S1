import math
import unittest
from types import SimpleNamespace

import numpy as np

from gui360_long_horizon.data.divergence import ScreenIndex, audit_pack, calibrate_tau, delta, detect_t_star, make_screen_point
from gui360_long_horizon.recovery_oracle import audit_precision, recovery_oracle


def _step(exec_id="e", step_id=1, controls=None, vec=None, total_steps=3, raw=None):
    raw = dict(raw or {})
    if vec is not None:
        raw["vec"] = np.asarray(vec, dtype=np.float32)
    return SimpleNamespace(
        exec_id=exec_id,
        step_id=step_id,
        total_steps=total_steps,
        request="Do task",
        template="template",
        app="excel",
        tag="in_app",
        screenshot_clean=f"{exec_id}_{step_id}.png",
        image_rel_path=f"test/image/{exec_id}_{step_id}.png",
        control_infos={"uia_controls_info": controls or []},
        raw=raw,
    )


def _embed(step, repo):
    return step.raw["vec"]


class DivergenceRecoveryTests(unittest.TestCase):
    def test_detect_t_star_and_delta(self):
        ok = [{"control_type": "Button", "control_text": "OK"}]
        success = [_step("s", 1, ok, [1, 0]), _step("s", 2, ok, [0, 1])]
        index = ScreenIndex(task_key="task", points=[make_screen_point(step, image_embedder=_embed) for step in success], alpha=1.0, band_width=1.0)
        fail = [_step("f", 1, ok, [1, 0]), _step("f", 2, ok, [-1, 0])]
        t_star = detect_t_star(fail, index, tau=0.1, image_embedder=_embed)
        self.assertEqual(t_star, 2)
        self.assertIsNone(delta(fail[0], t_star))
        self.assertEqual(delta(fail[1], t_star), 0)

    def test_calibrate_tau_separates_labels(self):
        tau = calibrate_tau([(0.9, False), (0.8, False), (0.2, True), (0.1, True)])
        self.assertGreaterEqual(tau, 0.2)
        self.assertLessEqual(tau, 0.8)

    def test_audit_pack_enforces_agreement(self):
        with self.assertRaises(ValueError):
            audit_pack(labeler_agreement=0.5)
        bundle = audit_pack(labeler_agreement=0.9)
        self.assertEqual(bundle.items, tuple())

    def test_recovery_oracle_detects_error_popup(self):
        controls = [
            {"control_type": "Dialog", "control_text": "Error: invalid value"},
            {"control_type": "Button", "control_text": "OK", "control_rect": [10, 20, 30, 40]},
        ]
        target = recovery_oracle(_step(controls=controls))
        self.assertIsNotNone(target)
        self.assertEqual(target.kind, "error_popup")
        self.assertEqual(target.correct_action.function, "click")
        self.assertEqual(target.correct_action.xy, (20.0, 30.0))

    def test_recovery_oracle_avoids_normal_window_false_positive(self):
        controls = [
            {"control_type": "Window", "control_text": "Excel"},
            {"control_type": "Button", "control_text": "Close", "control_rect": [0, 0, 1, 1]},
        ]
        self.assertIsNone(recovery_oracle(_step(controls=controls)))

    def test_audit_precision_threshold(self):
        positive = _step(controls=[{"control_type": "Dialog", "control_text": "Error"}, {"control_type": "Button", "control_text": "OK"}])
        normal = _step(controls=[{"control_type": "Window", "control_text": "Excel"}])
        self.assertTrue(math.isclose(audit_precision([(positive, True), (normal, False)]), 1.0))


if __name__ == "__main__":
    unittest.main()
