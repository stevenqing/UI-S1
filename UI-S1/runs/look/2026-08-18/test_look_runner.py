import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import run_look as runner


class LookRunnerTest(unittest.TestCase):
    def test_inputs_are_label_free(self):
        formal = runner.read_jsonl(runner.FORMAL_PATH)
        smoke = runner.read_jsonl(runner.SMOKE_PATH)
        runner.validate_rows(formal, 430)
        runner.validate_rows(smoke, 3)
        self.assertFalse({row["row_id"] for row in formal} & {row["row_id"] for row in smoke})

    def test_refuses_without_authorization(self):
        if runner.AUTHORIZATION_PATH.exists():
            self.skipTest("authorization exists")
        with self.assertRaises(PermissionError):
            runner.authorization()


if __name__ == "__main__":
    unittest.main()