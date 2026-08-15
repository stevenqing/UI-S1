import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_preflight", RUN_DIR / "prepare_preflight.py")
prepare_preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_preflight
SPEC.loader.exec_module(prepare_preflight)


class PreflightTest(unittest.TestCase):
    def test_forbidden_public_key(self):
        self.assertTrue(prepare_preflight.contains_forbidden_key({"nested": {"target_bbox": [0, 1]}}))
        self.assertFalse(prepare_preflight.contains_forbidden_key({"coordinate": [0, 1], "parse_ok": True}))

    def test_canonical_slot_order(self):
        self.assertEqual(prepare_preflight.CANONICAL_SLOTS[0], ("GTA1-7B", 0))
        self.assertEqual(prepare_preflight.CANONICAL_SLOTS[3], ("GTA1-7B", 1))
        self.assertEqual(prepare_preflight.CANONICAL_SLOTS[-1], ("UI-TARS-7B-SFT", 3))


if __name__ == "__main__":
    unittest.main()