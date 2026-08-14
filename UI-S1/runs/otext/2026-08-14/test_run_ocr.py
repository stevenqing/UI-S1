import unittest

from run_ocr import sha256_file


class OTEXTOCRWriterTest(unittest.TestCase):
    def test_locked_parameter_hash_shape(self):
        from run_ocr import ORTH_PREFLIGHT_SHA256
        self.assertEqual(len(ORTH_PREFLIGHT_SHA256), 64)


if __name__ == "__main__":
    unittest.main()