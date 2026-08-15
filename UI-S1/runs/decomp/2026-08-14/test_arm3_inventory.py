import importlib.util
import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("arm3_inventory", RUN_DIR / "arm3_inventory.py")
arm3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arm3
SPEC.loader.exec_module(arm3)


class Arm3InventoryTest(unittest.TestCase):
    def test_generation_logprob_detection(self):
        keys = arm3.recursive_keys({"generation": {"token_logprobs": [-1.0]}})
        self.assertTrue(arm3.normalized_leaf_keys(keys) & arm3.GENERATION_KEYS)

    def test_selector_logits_are_separate(self):
        leaves = arm3.normalized_leaf_keys(arm3.recursive_keys({"label_logits": [1.0]}))
        self.assertFalse(leaves & arm3.GENERATION_KEYS)
        self.assertTrue(leaves & arm3.SELECTOR_KEYS)

    def test_nested_list_schema(self):
        keys = arm3.recursive_keys({"outputs": [{"sequence_score": -2.0}]})
        self.assertIn("outputs[].sequence_score", keys)


if __name__ == "__main__":
    unittest.main()