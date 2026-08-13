import sys
import unittest
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from sequential_real_data_smoke import CONFIG_PATH, load_config


class SequentialRealDataSmokeTest(unittest.TestCase):
    def test_frozen_cross_fit_and_budget_contract(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        self.assertTrue(config["cross_fitting"]["verifier_inputs_must_be_cheap_oof"])
        self.assertTrue(config["cross_fitting"]["calibration_inputs_must_be_verifier_oof"])
        self.assertEqual(config["sequential_policy"]["budget_grid"]["androidcontrol"], [1, 2, 3])
        self.assertEqual(config["sequential_policy"]["budget_grid"]["mind2web"], [1, 2, 3, 4, 5, 6])
        self.assertFalse(config["execution"]["real_data_optimizer_authorized"])

    def test_smoke_source_cannot_construct_optimizer_or_backward(self):
        source = (RUN_DIR / "sequential_real_data_smoke.py").read_text()
        for forbidden in ("torch.optim", ".backward(", "optimizer.step("):
            self.assertNotIn(forbidden, source)

    def test_config_loads_with_bound_dependencies(self):
        self.assertEqual(
            load_config()["status"],
            "FROZEN_BEFORE_SEQUENTIAL_REAL_DATA_OPTIMIZER",
        )


if __name__ == "__main__":
    unittest.main()