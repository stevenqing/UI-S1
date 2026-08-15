import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("q1", RUN_DIR / "q1.py")
q1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = q1
SPEC.loader.exec_module(q1)


class Q1Test(unittest.TestCase):
    def test_summary(self):
        result = q1.summarize(Counter({"a": 1, "b": 1, "c": 2, "d": 4}))
        self.assertEqual(result["rows"], 8)
        self.assertEqual(result["screens"], 4)
        self.assertEqual(result["rows_per_screen_median"], 1.5)
        self.assertEqual(result["singleton_screen_fraction"], 0.5)
        self.assertEqual(result["rows_on_singleton_screen_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()