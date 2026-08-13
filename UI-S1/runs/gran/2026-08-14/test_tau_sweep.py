import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from gran_common import GranCandidate, tau_options
from run_tau_sweep import choose_tau, nested_primary


def make_rows():
    rows = {}
    for index in range(20):
        fold = index % 5
        candidates = (
            GranCandidate(
                source="a", lineage="a", action="POINT",
                coordinate=(0.0, 0.0), parameter="", parse_ok=True,
                order=0, correct=index % 2 == 0,
            ),
            GranCandidate(
                source="b", lineage="b", action="POINT",
                coordinate=(1.0, 0.0), parameter="", parse_ok=True,
                order=1, correct=index % 2 == 1,
            ),
        )
        rows[f"row-{index:02d}"] = {
            "fold": fold,
            "group": f"group-{index}",
            "candidates": candidates,
        }
    return rows


class TauSweepTest(unittest.TestCase):
    def test_exact_wins_validation_tie_by_fixed_order(self):
        rows = make_rows()
        options = tau_options([0.1, 0.2])
        selected, scores = choose_tau(
            rows,
            list(rows)[:10],
            list(rows)[10:],
            "screenspot_pro",
            options,
        )
        self.assertEqual(selected, ("exact", None))
        self.assertEqual(len(scores), 4)

    def test_nested_outputs_cover_each_row_once(self):
        rows = make_rows()
        result = nested_primary(
            rows, "screenspot_pro", tau_options([0.1, 0.2])
        )
        self.assertEqual(result["rows"], 20)
        self.assertEqual(set(result["mechanisms"]), set(rows))
        self.assertEqual(len(result["folds"]), 5)
        self.assertTrue(all(
            fold["selected_tau"] == "exact" for fold in result["folds"]
        ))
        self.assertTrue(all(
            not fold["finite_boundary_selected"] for fold in result["folds"]
        ))


if __name__ == "__main__":
    unittest.main()