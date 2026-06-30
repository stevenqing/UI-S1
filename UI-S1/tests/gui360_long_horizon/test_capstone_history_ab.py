import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from gui360_long_horizon.analysis.guards import FormatMismatchError, V1UtilizationClaimError
from gui360_long_horizon.analysis.capstone_report import aggregate_probe_summaries, load_results, write_results, write_verdict
from gui360_long_horizon.analysis.stats import capstone_decision, decision
from gui360_long_horizon.data.longdep_candidates import build_candidates, summary_for_pairs
from gui360_long_horizon.data.longdep_pairs import DifficultyBalanceError, LongDepPair, ShuffleControlError, assert_difficulty_balanced, assert_shuffle_collapse, make_shuffle_control, read_pair_file, write_pair_file
from gui360_long_horizon.experiments.capstone_runtime import action_match, load_turn_records, score_probe, score_v3_pairs, summarize_probe
from gui360_long_horizon.experiments import v1_ood_repair, v2_conditionC_matched, v3_longdep_pairs, v4_plan_recovery


def _row(step_uid, correct, **cond):
    return {"step_uid": step_uid, "ok": True, "correct": correct, "step_correct": correct, "cond": cond}


def _png(path: Path) -> None:
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path)


class _FakeModel:
    cache_id = "fake"

    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return [SimpleNamespace(text=self.text)]


class CapstoneHistoryABTests(unittest.TestCase):
    def test_v1_repairs_but_cannot_claim_utilization(self):
        rows = [_row("a", True, history_format="none"), _row("a", True, history_format="gt_history")]
        result = v1_ood_repair.summarize(rows, arm="G", history_format="gt_history")
        self.assertTrue(result.repaired)
        with self.assertRaises(V1UtilizationClaimError):
            v1_ood_repair.assert_v1_not_utilization()

    def test_format_mismatch_raises_for_matched_probes(self):
        rows = [_row("a", True, history_format="none", injected_error=0)]
        with self.assertRaises(FormatMismatchError):
            v2_conditionC_matched.summarize(rows, arm="G", history_format="none")

    def test_v2_and_v4_summaries(self):
        v2_rows = [
            _row("a", True, history_format="own_history", injected_error=0),
            _row("a", False, history_format="own_history", injected_error=1),
        ]
        v2 = v2_conditionC_matched.summarize(v2_rows, arm="O", history_format="own_history")
        self.assertEqual(v2.injected_minus_clean, -1.0)
        v4_rows = [
            _row("a", False, history_format="gt_history", plan="none"),
            _row("a", True, history_format="gt_history", plan="oracle"),
        ]
        v4 = v4_plan_recovery.summarize(v4_rows, arm="G", history_format="gt_history")
        self.assertEqual(v4.oracle_minus_none, 1.0)

    def test_longdep_pair_controls(self):
        pairs = [
            LongDepPair("p1", "n1", "f1", "dep_a", 2, 2, 5),
            LongDepPair("p2", "n2", "f2", "dep_b", 1, 1, 7),
        ]
        assert_difficulty_balanced(pairs)
        shuffled = make_shuffle_control(pairs)
        self.assertTrue(all(pair.shuffled for pair in shuffled))
        assert_shuffle_collapse(0.2, 0.0)
        with self.assertRaises(DifficultyBalanceError):
            assert_difficulty_balanced([LongDepPair("bad", "n", "f", "dep", 0, 2, 4)])
        with self.assertRaises(ShuffleControlError):
            assert_shuffle_collapse(0.2, 0.2)

    def test_longdep_candidate_builder_is_conservative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "data.json"
            dataset.write_text(json.dumps([
                {
                    "images": ["0.png", "1.png", "2.png", "3.png", "4.png"],
                    "conversations": [
                        {"from": "human", "value": "<image>\nEnter value"},
                        {"from": "gpt", "value": '<tool_call>{"function":"type","args":{"text":"alpha42"}}</tool_call>'},
                        {"from": "human", "value": "<image>\nUse alpha42 near"},
                        {"from": "gpt", "value": '<tool_call>{"function":"click","args":{"coordinate":[10,20]}}</tool_call>'},
                        {"from": "human", "value": "<image>\nMiddle filler"},
                        {"from": "gpt", "value": '<tool_call>{"function":"type","args":{"text":"filler"}}</tool_call>'},
                        {"from": "human", "value": "<image>\nStill about alpha42 far"},
                        {"from": "gpt", "value": '<tool_call>{"function":"click","args":{"coordinate":[30,40]}}</tool_call>'},
                    ],
                }
            ]), encoding="utf-8")
            pairs = build_candidates(dataset, min_distance_gap=2)
            pair_path = root / "pairs.json"
            write_pair_file(pairs, pair_path)
            loaded = read_pair_file(pair_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].dependency_key, "alpha42")
        self.assertEqual(loaded[0].near_d_bucket, loaded[0].far_d_bucket)
        self.assertFalse(summary_for_pairs(loaded, min_pairs=2)["v3_identifiable"])

    def test_v3_summary_requires_balanced_and_shuffle_clean(self):
        pairs = [LongDepPair("p1", "n1", "f1", "dep_a", 2, 2, 5), LongDepPair("p2", "n2", "f2", "dep_b", 2, 2, 5)]
        rows = [_row("n1", True, history_format="gt_history", distance="near"), _row("f1", False, history_format="gt_history", distance="far")]
        result = v3_longdep_pairs.summarize(rows, arm="G", history_format="gt_history", pairs=pairs, shuffle_gap=0.0)
        self.assertEqual(result.near_minus_far, 1.0)
        self.assertTrue(result.shuffle_clean)

    def test_capstone_runtime_loads_and_scores_sharegpt_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img0 = root / "0.png"
            img1 = root / "1.png"
            _png(img0)
            _png(img1)
            dataset = root / "data.json"
            dataset.write_text(json.dumps([
                {
                    "images": [str(img0), str(img1)],
                    "conversations": [
                        {"from": "human", "value": "<image>\nStep one"},
                        {"from": "gpt", "value": '<tool_call>{"function":"click","args":{"coordinate":[10,20]}}</tool_call>'},
                        {"from": "human", "value": "<image>\nStep two"},
                        {"from": "gpt", "value": '<tool_call>{"function":"click","args":{"coordinate":[30,40]}}</tool_call>'},
                    ],
                }
            ]), encoding="utf-8")
            records = load_turn_records(dataset)
            model = _FakeModel('<tool_call>{"function":"click","args":{"coordinate":[10,20]}}</tool_call>')
            rows = score_probe(records[:1], model, arm="G", history_format="gt_history", probe="v1")
            summary = summarize_probe(rows, probe="v1", arm="G", history_format="gt_history")
        self.assertEqual(len(records), 2)
        self.assertEqual(len(records[1].matched_images), 2)
        self.assertEqual(len(records[1].none_images), 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["correct"] for row in rows))
        self.assertTrue(summary["repaired"])
        self.assertGreaterEqual(len(model.calls), 2)

    def test_capstone_runtime_action_match(self):
        target = {"function": "click", "coordinate": [10, 20]}
        self.assertTrue(action_match({"function": "tap", "coordinate": [12, 18]}, target, coord_tol=5))
        self.assertFalse(action_match({"function": "type", "coordinate": [10, 20]}, target, coord_tol=5))
        type_target = {"function": "type", "text": "alpha42", "coordinate": [10, 20]}
        self.assertFalse(action_match({"function": "type", "text": "wrong", "coordinate": [10, 20]}, type_target, coord_tol=5))
        self.assertTrue(action_match({"function": "type", "text": "alpha42", "coordinate": [10, 20]}, type_target, coord_tol=5))

    def test_capstone_runtime_scores_v3_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images = []
            for idx in range(4):
                path = root / f"{idx}.png"
                _png(path)
                images.append(str(path))
            dataset = root / "data.json"
            dataset.write_text(json.dumps([
                {
                    "images": images,
                    "conversations": [
                        {"from": "human", "value": "<image>\nEnter alpha42"},
                        {"from": "gpt", "value": '<tool_call>{"function":"type","args":{"text":"alpha42"}}</tool_call>'},
                        {"from": "human", "value": "<image>\nUse alpha42 near"},
                        {"from": "gpt", "value": '<tool_call>{"function":"click","args":{"coordinate":[10,20]}}</tool_call>'},
                        {"from": "human", "value": "<image>\nFiller"},
                        {"from": "gpt", "value": '<tool_call>{"function":"type","args":{"text":"noop"}}</tool_call>'},
                        {"from": "human", "value": "<image>\nUse alpha42 far"},
                        {"from": "gpt", "value": '<tool_call>{"function":"click","args":{"coordinate":[10,20]}}</tool_call>'},
                    ],
                }
            ]), encoding="utf-8")
            records = load_turn_records(dataset)
            pair = LongDepPair("p", "ex0:turn1", "ex0:turn3", "alpha42", 0, 0, 2)
            model = _FakeModel('<tool_call>{"function":"click","args":{"coordinate":[10,20]}}</tool_call>')
            rows, summary = score_v3_pairs(records, model, arm="G", history_format="gt_history", pairs=[pair])
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["near_minus_far"], 0.0)
        self.assertTrue(summary["shuffle_clean"])

    def test_capstone_decision_rows(self):
        self.assertEqual(capstone_decision({"v1": {"repaired": False}}).label, "NOT_REPAIRED_ESCALATE")
        memory = capstone_decision({"v1": {"repaired": True}, "v3": {"near_minus_far": 0.08, "shuffle_clean": True}})
        self.assertEqual(memory.label, "REAL_HORIZON_MEMORY")
        strong_null = capstone_decision({"v1": {"repaired": True}, "v3": {"near_minus_far": 0.0, "shuffle_clean": True}, "v2": {"injected_minus_clean": 0.0}, "v4": {"oracle_minus_none": 0.13}})
        self.assertEqual(strong_null.label, "STRONG_NULL_MEMORY")
        separate = decision({"v1": {"repaired": True}, "v3": {"near_minus_far": 0.0, "shuffle_clean": True}, "v2": {"injected_minus_clean": 0.0}, "v4": {"oracle_minus_none": 0.02}})
        self.assertEqual(separate.label, "PLAN_HISTORY_SEPARATE_DEFICITS")

    def test_write_verdict_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_path = root / "results.json"
            verdict_path = root / "verdict.json"
            results = {"v1": {"repaired": False}}
            results_path.write_text(json.dumps(results), encoding="utf-8")
            self.assertEqual(load_results(results_path), results)
            write_verdict(results, verdict_path)
            payload = json.loads(verdict_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["verdict"]["label"], "NOT_REPAIRED_ESCALATE")

    def test_aggregate_probe_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summaries = root / "summaries"
            summaries.mkdir()
            (summaries / "G_v1.json").write_text(json.dumps({"repaired": True, "matched_minus_none": 0.01}), encoding="utf-8")
            (summaries / "O_v1.json").write_text(json.dumps({"repaired": True, "matched_minus_none": 0.03}), encoding="utf-8")
            (summaries / "O_v2.json").write_text(json.dumps({"injected_minus_clean": 0.0}), encoding="utf-8")
            (summaries / "O_v4.json").write_text(json.dumps({"oracle_minus_none": 0.13}), encoding="utf-8")
            v3 = root / "v3.json"
            v3.write_text(json.dumps({"near_minus_far": 0.0, "shuffle_clean": True}), encoding="utf-8")
            results = aggregate_probe_summaries(summaries, v3_summary=v3)
            out = root / "results.json"
            write_results(results, out)
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(payload["v1"]["repaired"])
        self.assertEqual(payload["v1"]["required_arms"], ["G", "O"])
        self.assertAlmostEqual(payload["og_contrast"]["value"], 0.02)
        self.assertEqual(capstone_decision(payload).label, "STRONG_NULL_MEMORY")


if __name__ == "__main__":
    unittest.main()