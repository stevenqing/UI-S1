import json
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import run_arm_a as runner


class ArmARunnerTest(unittest.TestCase):
    def test_coordinate_transform_matches_h1_crop_semantics(self):
        local, full = runner.map_coordinate([100, 200], [1288, 728], [2576, 1456], True, [30, 40])
        self.assertEqual(local, [100, 200])
        self.assertEqual(full, [130, 240])

    def test_parser_matches_h1_integer_contract(self):
        extractor = lambda value: tuple(map(int, __import__("re").search(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", value).groups()))
        self.assertEqual(runner.parse_output("(12,34)", extractor)["parse_status"], "parsed")
        self.assertEqual(runner.parse_output("(12.5,34)", extractor)["parse_status"], "unparsable")

    def test_trace_schema_rejects_evaluation_field(self):
        row = {field: None for field in runner.REQUIRED_TRACE_FIELDS}
        row["target_bbox"] = [0, 0, 1, 1]
        with self.assertRaises(ValueError):
            runner.validate_trace_row(row)

    def test_input_manifests_are_label_free_and_disjoint(self):
        formal = runner.read_jsonl(runner.FORMAL_INPUT_PATH)
        smoke = runner.read_jsonl(runner.SMOKE_INPUT_PATH)
        runner.validate_input_rows(formal, "formal")
        runner.validate_input_rows(smoke, "smoke")
        self.assertFalse({row["row_id"] for row in formal} & {row["row_id"] for row in smoke})

    def test_formal_shard_boundaries(self):
        rows = runner.read_jsonl(runner.FORMAL_INPUT_PATH)
        counts = {shard: sum(row["execution_shard"] == shard for row in rows) for shard in runner.FORMAL_SHARDS}
        self.assertEqual(counts, {"common_11": 200, "partial_1_10": 150, "uncovered_0": 150})

    def test_runner_refuses_gpu_without_authorization(self):
        if runner.AUTHORIZATION_PATH.exists():
            self.skipTest("authorization exists")
        with self.assertRaises(PermissionError):
            runner.load_authorization()


if __name__ == "__main__":
    unittest.main()