import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from r1_headroom import gate_pass, load_locked_recovery, load_r1_config, majority_candidate
from recovery_common import identity_hash


class R1Test(unittest.TestCase):
    def test_r1_is_sealed_without_recovery_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                load_locked_recovery(Path(directory) / "RECOVERY_MANIFEST.json")

    def test_frozen_r1_config(self):
        config = load_r1_config()
        self.assertEqual(config["practical_headroom_margin"], 0.01)
        self.assertEqual(config["models"], ["UI-AGILE-7B", "GUI-R1-7B", "UI-R1-E-3B"])

    def test_majority_uses_action_plurality_then_priority(self):
        candidates = [
            {"source": "A", "action": "click", "parse_ok": True},
            {"source": "B", "action": "type", "parse_ok": True},
            {"source": "C", "action": "type", "parse_ok": True},
        ]
        self.assertEqual(majority_candidate(candidates, ["A", "B", "C"])["source"], "B")
        tied = [
            {"source": "A", "action": "click", "parse_ok": True},
            {"source": "B", "action": "type", "parse_ok": True},
        ]
        self.assertEqual(majority_candidate(tied, ["B", "A"])["source"], "B")

    def test_gate_requires_margin_and_positive_lower_bound(self):
        self.assertTrue(gate_pass({"point_delta": 0.011, "ci_99": [0.001, 0.02]}, 0.01))
        self.assertFalse(gate_pass({"point_delta": 0.009, "ci_99": [0.001, 0.02]}, 0.01))
        self.assertFalse(gate_pass({"point_delta": 0.02, "ci_99": [-0.001, 0.03]}, 0.01))

    def test_manifest_guard_precedes_evaluator_import_and_gt_access(self):
        source = (Path(__file__).resolve().parent / "r1_headroom.py").read_text()
        main = next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = [
            node.func.id for statement in main.body for node in ast.walk(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertLess(calls.index("load_locked_recovery"), calls.index("load_scoring_and_bootstrap"))
        self.assertLess(calls.index("load_locked_recovery"), calls.index("analyze_setting"))

    def test_identity_hash_is_order_invariant_but_identity_sensitive(self):
        rows = [
            {"stable_index": 1, "id": "second"},
            {"stable_index": 0, "id": "first"},
        ]
        expected = identity_hash(rows)
        self.assertEqual(expected, identity_hash(list(reversed(rows))))
        changed = [dict(row) for row in rows]
        changed[0]["id"] = "changed"
        self.assertNotEqual(expected, identity_hash(changed))

    def test_manifest_guard_revalidates_rows_bytes_and_identity(self):
        source = (Path(__file__).resolve().parent / "r1_headroom.py").read_text()
        function = next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "load_locked_recovery"
        )
        calls = {
            node.func.id for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue({"load_jsonl", "validate_lane_rows", "identity_hash", "sha256_file"} <= calls)


if __name__ == "__main__":
    unittest.main()