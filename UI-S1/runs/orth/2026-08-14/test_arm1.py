import unittest

from arm1_ocr_analysis import (
    edit_similarity, match_box, normalize_text, prepare_boxes, select_prepared,
)


class OCRAnalysisTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_text("  SAVE! "), "save")

    def test_edit_similarity(self):
        self.assertGreater(edit_similarity("setings", "open settings menu"), 0.8)

    def test_match_tie_prefers_confidence_after_score_and_length(self):
        boxes = [
            {"text": "Save", "confidence": 0.5, "polygon": [[0,0],[1,0],[1,1],[0,1]], "engine_order": 0},
            {"text": "Save", "confidence": 0.9, "polygon": [[2,0],[3,0],[3,1],[2,1]], "engine_order": 1},
        ]
        self.assertEqual(match_box(boxes, "Click Save", "normalized", 1)["engine_order"], 1)
        prepared = prepare_boxes(boxes, "Click Save")
        self.assertEqual(select_prepared(prepared, "normalized", 1)["engine_order"], 1)


if __name__ == "__main__":
    unittest.main()