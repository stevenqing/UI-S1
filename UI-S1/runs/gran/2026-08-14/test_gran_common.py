import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from gran_common import (
    GranCandidate, density_select, mechanism_values, prior_select, tau_options,
)


def candidate(source, coordinate, correct, reliability, order, action="CLICK"):
    return GranCandidate(
        source=source,
        lineage=source,
        action=action,
        coordinate=coordinate,
        parameter="",
        parse_ok=True,
        order=order,
        correct=correct,
        reliability=reliability,
    )


class GranCommonTest(unittest.TestCase):
    def test_exact_and_single_endpoints_equal_prior_with_prior_pi(self):
        rows = (
            candidate("a", (0.0, 0.0), False, 0.9, 0),
            candidate("b", (1.0, 0.0), True, 0.8, 1),
            candidate("c", (2.0, 0.0), True, 0.7, 2),
        )
        prior = prior_select(rows)
        exact, _ = density_select(rows, "screenspot_pro", "exact")
        single, _ = density_select(rows, "screenspot_pro", "single")
        self.assertEqual(exact.source, prior.source)
        self.assertEqual(single.source, prior.source)

    def test_density_count_can_beat_prior(self):
        rows = (
            candidate("a", (0.0, 0.0), False, 0.9, 0),
            candidate("b", (1.0, 0.0), True, 0.8, 1),
            candidate("c", (1.01, 0.0), True, 0.7, 2),
        )
        selected, details = density_select(
            rows, "screenspot_pro", "finite", 0.02
        )
        self.assertTrue(selected.correct)
        self.assertEqual(details["votes"], 2)

    def test_mind2web_is_type_first(self):
        rows = (
            candidate("a", (0.0, 0.0), False, 0.9, 0, "CLICK"),
            candidate("b", (0.0, 0.0), True, 0.8, 1, "TYPE"),
        )
        selected, details = density_select(rows, "mind2web", "single")
        self.assertEqual(details["votes"], 1)
        self.assertEqual(selected.source, "a")

    def test_mechanism_values(self):
        rows = (
            candidate("a", (0.0, 0.0), True, 0.9, 0),
            candidate("b", (0.01, 0.0), True, 0.8, 1),
            candidate("c", (1.0, 0.0), False, 0.7, 2),
        )
        values = mechanism_values(rows, "screenspot_pro", "finite", 0.02)
        self.assertAlmostEqual(values["p_hat"], 2 / 3)
        self.assertEqual(values["q_max_hat"], 1.0)
        self.assertEqual(values["contamination"], 0.0)

    def test_tau_order_is_exact_finite_single(self):
        self.assertEqual(
            tau_options([0.1, 0.2]),
            (("exact", None), ("finite", 0.1), ("finite", 0.2), ("single", None)),
        )


if __name__ == "__main__":
    unittest.main()