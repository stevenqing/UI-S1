import unittest

from run_ocr import normalize_polygon


class OCRWriterTest(unittest.TestCase):
    def test_polygon_serialization(self):
        value = normalize_polygon([[1, 2], [3, 2], [3, 4], [1, 4]])
        self.assertEqual(value, [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])

    def test_rejects_invalid_polygon(self):
        with self.assertRaises(ValueError):
            normalize_polygon([[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()