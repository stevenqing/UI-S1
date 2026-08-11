import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_data import baseline_vus_budget, evidence_prompt, render_evidence


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"


class EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [json.loads(line) for line in (VUS / "data/public_records.jsonl").read_text().splitlines() if line.strip()]
        cls.record = next(row for row in rows if row["benchmark"] == "screenspot_pro" and row["arm"] == "C_uni")

    def test_all_modes_respect_pixel_budget(self):
        for mode in ("local", "random", "global_only", "fine_only", "context_only"):
            images, permutation, prompt, budget = render_evidence(self.record, mode)
            self.assertEqual(sorted(permutation), list(range(12)))
            self.assertLessEqual(budget["expected_pixel_ratio_vs_vus"], 1.02)
            self.assertGreaterEqual(budget["expected_pixel_ratio_vs_vus"], 0.90)
            for actual, target in zip(budget["expected_processed_pixels"], budget["target_pixels"]):
                self.assertGreaterEqual(actual / target, 0.85)
            self.assertEqual(sum(image.width * image.height for image in images), budget["expected_total_processed_pixels"])
            self.assertNotIn("target_bbox", prompt.lower())
            self.assertNotIn("fallback", prompt.lower())

    def test_local_and_random_have_identical_budget_and_prompt(self):
        local_images, local_permutation, local_prompt, local_budget = render_evidence(self.record, "local")
        random_images, random_permutation, random_prompt, random_budget = render_evidence(self.record, "random")
        self.assertEqual(local_permutation, random_permutation)
        self.assertEqual(local_prompt, random_prompt)
        self.assertEqual(local_budget["image_dimensions"], random_budget["image_dimensions"])
        self.assertNotEqual(local_images[1].tobytes(), random_images[1].tobytes())

    def test_ultrawide_budget_is_feasible(self):
        rows = [json.loads(line) for line in (VUS / "data/public_records.jsonl").read_text().splitlines() if line.strip()]
        record = next(row for row in rows if row["row_id"] == "autocad_windows_0" and row["arm"] == "C_uni")
        _, _, _, budget = render_evidence(record, "local")
        self.assertEqual(budget["baseline_vus_pixels"], 716800)
        self.assertLessEqual(budget["expected_pixel_ratio_vs_vus"], 1.02)


if __name__ == "__main__":
    unittest.main()
