import ast
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_private_labels import build_labels, load_config, score_candidate


class FakeScoring:
    GROUNDING_ACTIONS = {"click"}
    TEXT_ACTIONS = {"type", "scroll"}
    SIMPLE_ACTIONS = {"press_back"}

    @staticmethod
    def text_f1(left, right):
        return float(left.lower() == right.lower())


class PrivateLabelTest(unittest.TestCase):
    def reference(self, action="click"):
        return {
            "gt_action": action,
            "gt_bbox": [20, 30],
            "gt_input_text": "Hello",
            "image_size": [100, 100],
        }

    def test_frozen_config_and_committed_blind_lock(self):
        config = load_config()
        self.assertEqual(config["expected_fold_rows"], {0: 792, 1: 754, 2: 826, 3: 870, 4: 758})
        self.assertEqual(config["candidate_count"], 3)

    def test_candidate_scoring_contract(self):
        config = load_config()
        click = {"action": "click", "coordinate": [0.2, 0.3], "parameter": "", "parse_ok": True}
        self.assertTrue(score_candidate(self.reference(), click, FakeScoring, config))
        far = dict(click, coordinate=[0.8, 0.9])
        self.assertFalse(score_candidate(self.reference(), far, FakeScoring, config))
        boundary = dict(click, coordinate=[0.34, 0.3])
        self.assertFalse(score_candidate(self.reference(), boundary, FakeScoring, config))
        typed = {"action": "type", "coordinate": None, "parameter": "hello", "parse_ok": True}
        self.assertTrue(score_candidate(self.reference("type"), typed, FakeScoring, config))
        scroll = {"action": "scroll", "coordinate": None, "parameter": "DOWN", "parse_ok": True}
        scroll_reference = self.reference("scroll")
        scroll_reference["gt_input_text"] = "down"
        self.assertTrue(score_candidate(scroll_reference, scroll, FakeScoring, config))
        back = {"action": "press_back", "coordinate": None, "parameter": "", "parse_ok": True}
        self.assertTrue(score_candidate(self.reference("press_back"), back, FakeScoring, config))
        self.assertFalse(score_candidate(self.reference(), dict(click, parse_ok=False), FakeScoring, config))
        unknown = {"action": "UNKNOWN", "coordinate": None, "parameter": "", "parse_ok": True}
        self.assertFalse(score_candidate(self.reference("UNKNOWN"), unknown, FakeScoring, config))

    def test_private_output_schema_is_minimal(self):
        source = (Path(__file__).resolve().parent / "prepare_private_labels.py").read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_labels")
        literal_keys = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Dict):
                keys = {
                    key.value for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if "candidate_success" in keys:
                    literal_keys = keys
        self.assertEqual(literal_keys, {"schema_version", "sample_key", "candidate_success"})

    def test_builder_enforces_fold_range_and_counts(self):
        source = (Path(__file__).resolve().parent / "prepare_private_labels.py").read_text()
        self.assertIn('output[row["fold"]]', source)
        self.assertIn('config["expected_fold_rows"][fold]', source)

    def test_builder_scores_serialized_public_candidate_order(self):
        source = (Path(__file__).resolve().parent / "prepare_private_labels.py").read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_labels")
        source_text = ast.unparse(function)
        self.assertIn('for candidate in row[\'candidates\']', source_text)
        self.assertNotIn("canonical_private_source_order", source_text)

    def test_blind_guard_precedes_scorer_and_gt_access(self):
        source = (Path(__file__).resolve().parent / "prepare_private_labels.py").read_text()
        tree = ast.parse(source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [
            node.func.id for statement in main.body for node in ast.walk(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertLess(calls.index("load_config"), calls.index("load_scoring"))
        self.assertLess(calls.index("load_config"), calls.index("build_labels"))

    def test_manifest_code_contains_no_aggregate_metric(self):
        source = (Path(__file__).resolve().parent / "prepare_private_labels.py").read_text()
        tree = ast.parse(source)
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse({"mean", "average", "roc_auc_score", "paired_bootstrap"} & (attributes | names))


if __name__ == "__main__":
    unittest.main()