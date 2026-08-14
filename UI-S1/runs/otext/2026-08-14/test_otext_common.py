import unittest

from otext_common import best_ocr_box, extract_literals, weighted_b3


class OTEXTCommonTest(unittest.TestCase):
    def test_extractors(self):
        self.assertEqual(extract_literals('Click "Save As" now', 'quoted'), ['Save As'])
        self.assertEqual(extract_literals('Open API Settings', 'caps_camel'), ['API'])
        self.assertEqual(extract_literals('  Click Save! ', 'full_normalized'), ['click save'])

    def test_score_uses_confidence(self):
        boxes = [
            {"text": "Save", "confidence": 0.4, "polygon": [[0,0],[1,0],[1,1],[0,1]], "engine_order": 0},
            {"text": "Save", "confidence": 0.9, "polygon": [[2,0],[3,0],[3,1],[2,1]], "engine_order": 1},
        ]
        self.assertEqual(best_ocr_box(boxes, 'Click "Save"', 'quoted', 'normalized')["box"]["engine_order"], 1)

    def test_weighted_b3(self):
        selected, group = weighted_b3([(0,0),(1,1),(100,100)], [0.1,0.1,0.5])
        self.assertEqual(selected, 2)


if __name__ == "__main__":
    unittest.main()