import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from sequential_stopping_diagnostic import (
    apply_policy, fallback_reliability, select_parameters, summarize,
)


class SequentialStoppingDiagnosticTest(unittest.TestCase):
    @staticmethod
    def rows():
        return [
            {
                "sample_key": f"row-{index}",
                "cell": "cell",
                "order": [0, 1, 2],
                "cheap_probability": probabilities,
                "verifier_probability": probabilities,
            }
            for index, probabilities in enumerate((
                [0.9, 0.2, 0.1], [0.8, 0.3, 0.1],
                [0.2, 0.8, 0.1], [0.1, 0.9, 0.2],
            ))
        ]

    def test_policy_accepts_cross_fitted_candidate_or_fallback(self):
        rows = self.rows()
        labels = {
            "row-0": [True, False, False],
            "row-1": [True, False, False],
            "row-2": [False, True, False],
            "row-3": [False, True, False],
        }
        strongest = {key: False for key in labels}
        reliability = fallback_reliability(rows, strongest)
        outputs = apply_policy(
            rows, labels, strongest, reliability, "cheap", (2, 0.5, 0.2)
        )
        self.assertEqual(summarize(outputs)["cells"]["cell"]["accuracy"], 1.0)

    def test_parameter_selection_includes_safe_fallback(self):
        rows = self.rows()
        labels = {row["sample_key"]: [False, False, False] for row in rows}
        strongest = {key: True for key in labels}
        config = {
            "sequential_policy": {
                "budget_grid": {"mind2web": [1, 2]},
                "minimum_delta_grid": [0.0, 0.1],
                "maximum_loss_risk_grid": [0.0, 0.2],
            },
            "safety": {"mde": {"mind2web": 0.01}},
        }
        selected = select_parameters(
            rows, labels, strongest, "cheap", config, "mind2web"
        )
        self.assertIsNone(selected["parameters"])


if __name__ == "__main__":
    unittest.main()