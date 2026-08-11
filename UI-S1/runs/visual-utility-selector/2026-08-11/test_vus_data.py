import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vus_data import (
    CANDIDATE_LABELS,
    KEEP_LABEL,
    audit_public_record,
    build_candidate_prompt,
    build_prompt,
    deterministic_permutation,
    render_overlay,
    sha256_file,
    target_label,
)
from adjudicate_anchor import load_anchor_test_labels_after_pretest


class VUSDataTest(unittest.TestCase):
    def test_anchor_outer_labels_require_pretest_record(self):
        labels = Path(self.temporary.name) / "private_labels_fold-1.jsonl"
        labels.write_text(__import__("json").dumps({"sample_key": "key", "candidate_success": [False] * 12}) + "\n")
        pretest = Path(self.temporary.name) / "anchor_pretest.json"
        with self.assertRaises(PermissionError):
            load_anchor_test_labels_after_pretest(1, pretest, label_dir=Path(self.temporary.name))
        pretest.write_text(__import__("json").dumps({
            "status": "PASS_ALL_ANCHOR_SELECTIONS_FROZEN_BEFORE_OUTER_LABEL_ACCESS",
            "folds": [{"outer_fold": 1, "opened_development_label_folds": [0, 2, 3, 4]}],
        }))
        self.assertEqual(
            set(load_anchor_test_labels_after_pretest(1, pretest, label_dir=Path(self.temporary.name))),
            {"key"},
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary.name) / "screen.png"
        Image.new("RGB", (320, 180), "white").save(self.image_path)
        self.record = {
            "sample_key": "mind2web/C_uni/example",
            "benchmark": "mind2web",
            "arm": "C_uni",
            "row_id": "example",
            "fold": 0,
            "group": "episode",
            "image_path": str(self.image_path),
            "image_sha256": sha256_file(self.image_path),
            "instruction": "open settings",
            "history": ["CLICK menu"],
            "candidates": [
                {"action": "CLICK", "coordinate": [0.5, 0.5], "parameter": "", "parse_ok": True}
                for _ in range(12)
            ],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_permutation_is_deterministic_and_epoch_dependent(self):
        first = deterministic_permutation("key", 0, 11)
        self.assertEqual(first, deterministic_permutation("key", 0, 11))
        self.assertEqual(sorted(first), list(range(12)))
        self.assertNotEqual(first, deterministic_permutation("key", 1, 11))

    def test_target_keeps_correct_fallback(self):
        permutation = tuple(range(12))
        success = [False] * 12
        success[3] = True
        self.assertEqual(target_label(success, 3, permutation, "key", 0, 11), KEEP_LABEL)

    def test_target_chooses_positive_when_fallback_wrong(self):
        permutation = tuple(reversed(range(12)))
        success = [False] * 12
        success[3] = True
        label = target_label(success, 0, permutation, "key", 0, 11)
        self.assertEqual(label, CANDIDATE_LABELS[permutation.index(3)])

    def test_prompt_and_overlay_do_not_need_labels(self):
        permutation = tuple(range(12))
        prompt = build_prompt(self.record, 2, permutation)
        self.assertIn("M: KEEP_CEV (same action as C)", prompt)
        self.assertNotIn("correct", prompt.lower())
        overlay = render_overlay(self.record, permutation)
        self.assertEqual(overlay.size, (320, 180))

    def test_candidate_prompt_is_fallback_agnostic(self):
        prompt = build_candidate_prompt(self.record, tuple(range(12)))
        self.assertIn("A through L", prompt)
        self.assertNotIn("CEV", prompt)
        self.assertNotIn("KEEP", prompt)

    def test_out_of_frame_candidate_is_preserved_and_rendered(self):
        self.record["candidates"][0]["coordinate"] = [1.048, -0.02]
        prompt = build_prompt(self.record, 2, tuple(range(12)))
        self.assertIn("x=1.0480, y=-0.0200", prompt)
        overlay = render_overlay(self.record, tuple(range(12)))
        self.assertEqual(overlay.size, (320, 180))

    def test_public_audit_rejects_ground_truth(self):
        self.assertTrue(audit_public_record(self.record))
        leaked = dict(self.record, target_bbox=[1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "V-K1"):
            audit_public_record(leaked)


if __name__ == "__main__":
    unittest.main()
