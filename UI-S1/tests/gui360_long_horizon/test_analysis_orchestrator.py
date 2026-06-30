import unittest
from types import SimpleNamespace

from gui360_long_horizon.analysis.controls import shuffle_test
from gui360_long_horizon.analysis.identifiability import assert_no_causal_claim_from, label_output
from gui360_long_horizon.analysis.stats import DecisionAborted, bootstrap_ci, decision, existence_verdict
from gui360_long_horizon.run_all import GateFailed, load_config, run_stages
from gui360_long_horizon.types import Row


def _row(block, correct):
    return Row(
        exec_id=f"e-{block}-{correct}",
        app="excel",
        tag="in_app",
        task_key="task",
        step_id=1,
        depth_frac=0.1,
        manifold="on",
        delta=None,
        d_bucket=0,
        cond={"block": block},
        step_correct=correct,
        model="dummy",
    )


class AnalysisOrchestratorTests(unittest.TestCase):
    def test_identifiability_guard_blocks_diag_causal_claim(self):
        with self.assertRaises(RuntimeError):
            assert_no_causal_claim_from("m_diag")
        labeled = label_output("m_plan", {"ok": True})
        self.assertTrue(labeled.identified)

    def test_existence_verdict_ignores_nonidentified_position(self):
        result = SimpleNamespace(rows=[_row("base", False), _row("position", True), _row("plan", True)], blocks={"position": False, "plan": True})
        verdict = existence_verdict(result)
        self.assertTrue(verdict["exists"])
        self.assertFalse(verdict["effects"]["position"]["identified"])

    def test_decision_aborts_when_textmem_gate_fails(self):
        with self.assertRaises(DecisionAborted) as ctx:
            decision({"m_textmem": SimpleNamespace(gate_passed=False)})
        self.assertEqual(ctx.exception.verdict.label, "ABORT_TEXTMEM_GATE")

    def test_bootstrap_ci_and_shuffle_test(self):
        rows = [_row("base", False), _row("plan", True), _row("plan", True)]
        ci = bootstrap_ci(rows, lambda df: float(df["step_correct"].mean()), B=20)
        self.assertGreaterEqual(ci.high, ci.low)
        self.assertTrue(shuffle_test(lambda labels: sum(labels) - len(labels) / 2, [1, 0, 1, 0]))

    def test_run_stages_dry_run_and_gate_failure(self):
        config = load_config("gui360_long_horizon/configs/default.yaml")
        dry = run_stages(config, dry_run=True, stages=["loader_smoke", "textmem_gate"])
        self.assertEqual([stage.name for stage in dry], ["loader_smoke", "textmem_gate"])
        with self.assertRaises(GateFailed) as missing:
            run_stages(config, runners={}, stages=["loader_smoke"])
        self.assertTrue(missing.exception.result.skipped)
        with self.assertRaises(GateFailed) as ctx:
            run_stages(config, runners={"difficulty_gate": lambda cfg: {"passed": False}}, stages=["difficulty_gate", "diag_bound"])
        self.assertEqual(ctx.exception.result.name, "difficulty_gate")


if __name__ == "__main__":
    unittest.main()
