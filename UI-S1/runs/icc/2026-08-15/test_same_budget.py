import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("same_budget", RUN_DIR / "same_budget.py")
same_budget = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = same_budget
SPEC.loader.exec_module(same_budget)


class SameBudgetTest(unittest.TestCase):
    def test_pool_names(self):
        self.assertEqual(len(same_budget.POOL_NAMES), 4)
        self.assertEqual(set(same_budget.OMITTED.values()), {"GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT"})

    def test_historical_pool_is_omit_ui_tars(self):
        self.assertEqual(same_budget.OMITTED["gta1_qwen3_6x2"], "UI-TARS-7B-SFT")


if __name__ == "__main__":
    unittest.main()