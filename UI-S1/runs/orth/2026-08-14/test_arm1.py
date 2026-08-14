import unittest

from arm1_ocr_analysis import (
    edit_similarity, edit_similarity_prepared, match_box, normalize_text,
    prepare_boxes, select_prepared, summarize,
)


class OCRAnalysisTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text("  SAVE! "), "save")

    def test_edit_similarity(self):
        self.assertGreater(edit_similarity("setings", "open settings menu"), 0.8)
        self.assertEqual(
            edit_similarity("setings", "open settings menu"),
            edit_similarity_prepared("setings", ["open", "settings", "menu"]),
        )

    def test_match_tie_prefers_confidence_after_score_and_length(self):
        boxes = [
            {"text": "Save", "confidence": 0.5, "polygon": [[0,0],[1,0],[1,1],[0,1]], "engine_order": 0},
            {"text": "Save", "confidence": 0.9, "polygon": [[2,0],[3,0],[3,1],[2,1]], "engine_order": 1},
        ]
        self.assertEqual(match_box(boxes, "Click Save", "normalized", 1)["engine_order"], 1)
        prepared = prepare_boxes(boxes, "Click Save")
        self.assertEqual(select_prepared(prepared, "normalized", 1)["engine_order"], 1)

    def test_summary_is_structured(self):
        rows = [
            {"matched": True, "row_class": "selected_correct", "ui_type": "text", "correct": True, "pool_error": False, "matching_boxes": 1},
            {"matched": False, "row_class": "recoverable", "ui_type": "icon", "correct": False, "pool_error": True, "matching_boxes": 0},
            {"matched": False, "row_class": "zero_coverage", "ui_type": "icon", "correct": False, "pool_error": True, "matching_boxes": 0},
        ]
        report = summarize(rows)
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["class_match_table"]["selected_correct"]["matched"], 1)


if __name__ == "__main__":
    unittest.main()