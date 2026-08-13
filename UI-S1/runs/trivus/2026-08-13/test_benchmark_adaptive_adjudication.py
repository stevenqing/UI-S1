import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_adaptive_adjudication import diagnose


class BenchmarkAdaptiveAdjudicationTest(unittest.TestCase):
    @staticmethod
    def fixture(target=True, baseline=False, joint=False):
        public = {}
        for family, cells in {
            "mind2web": ("C_uni", "C_cond", "C_rand", "C_self"),
            "screenspot_pro": ("C_uni", "C_cond", "C_rand", "C_self"),
            "androidcontrol": ("low", "high"),
        }.items():
            for cell in cells:
                for index in range(10):
                    key = f"{family}/{cell}/{index}"
                    public[key] = {
                        "benchmark": family,
                        "arm": cell if family != "androidcontrol" else None,
                        "setting": cell if family == "androidcontrol" else None,
                        "fold": index % 5,
                        "group": f"{family}/{cell}/group-{index}",
                    }
        outputs = {
            "TARGET_ONLY": {
                "safe": {key: target for key in public},
                "direct": {key: target for key in public},
            },
            "JOINT3": {"safe": {key: joint for key in public}},
        }
        config = {
            "statistics": {
                "resamples": 100,
                "bootstrap_seed_base": 20260900,
            },
            "thresholds": {
                "mde": {
                    "mind2web": 0.01,
                    "screenspot_pro": 0.01,
                    "androidcontrol": 0.01,
                },
            },
        }
        baseline_values = {key: baseline for key in public}
        return outputs, public, baseline_values, config

    def test_each_benchmark_is_adjudicated_independently(self):
        outputs, public, baseline, config = self.fixture()
        result = diagnose(outputs, public, baseline, baseline, config)
        self.assertEqual(
            set(result), {"mind2web", "screenspot_pro", "androidcontrol"}
        )
        for value in result.values():
            self.assertTrue(value["gates"]["benchmark_ready_diagnostic"])

    def test_one_failed_benchmark_is_not_rescued_by_others(self):
        outputs, public, baseline, config = self.fixture()
        for key in outputs["TARGET_ONLY"]["safe"]:
            if public[key]["benchmark"] == "mind2web":
                outputs["TARGET_ONLY"]["safe"][key] = False
                baseline[key] = True
        result = diagnose(outputs, public, baseline, baseline, config)
        self.assertFalse(result["mind2web"]["gates"]["benchmark_ready_diagnostic"])
        self.assertTrue(result["screenspot_pro"]["gates"]["benchmark_ready_diagnostic"])
        self.assertTrue(result["androidcontrol"]["gates"]["benchmark_ready_diagnostic"])

    def test_incomplete_target_coverage_is_rejected(self):
        outputs, public, baseline, config = self.fixture()
        outputs["TARGET_ONLY"]["safe"].pop(next(iter(public)))
        with self.assertRaisesRegex(ValueError, "coverage"):
            diagnose(outputs, public, baseline, baseline, config)

    def test_oracle_headroom_uses_direct_or_strongest(self):
        outputs, public, strongest, config = self.fixture(
            target=False, baseline=False
        )
        keys = sorted(
            key for key, row in public.items()
            if row["benchmark"] == "androidcontrol"
        )
        for key in keys[::2]:
            outputs["TARGET_ONLY"]["direct"][key] = True
        result = diagnose(outputs, public, strongest, strongest, config)
        headroom = result["androidcontrol"]["incremental_utility_headroom"]
        self.assertGreater(headroom["point_delta"], 0)


if __name__ == "__main__":
    unittest.main()