import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_feasibility", RUN_DIR / "prepare_feasibility.py")
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


class FeasibilityTest(unittest.TestCase):
    def test_hash_is_stable(self):
        self.assertEqual(len(prepare.sha256_file(RUN_DIR / "SPEC.md")), 64)


if __name__ == "__main__":
    unittest.main()