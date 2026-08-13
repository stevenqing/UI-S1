import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

import sequential_authorization


class SequentialAuthorizationTest(unittest.TestCase):
    def test_helpers_import_when_authorization_path_exists(self):
        helpers = sequential_authorization._helpers()
        self.assertEqual(len(helpers), 4)

    def test_missing_authorization_fails_closed(self):
        original = sequential_authorization.AUTHORIZATION_PATH
        sequential_authorization.AUTHORIZATION_PATH = RUN_DIR / "missing-auth.json"
        try:
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                sequential_authorization.load_bound_authorization()
        finally:
            sequential_authorization.AUTHORIZATION_PATH = original

    def test_nonce_paths_are_hidden_and_canonical(self):
        nonce = "a" * 64
        self.assertTrue(sequential_authorization.attempt_path(nonce).name.startswith(".sequential-attempt-"))
        self.assertEqual(sequential_authorization.receipt_path(nonce).suffix, ".json")
        with self.assertRaises(PermissionError):
            sequential_authorization.attempt_path("../escape")


if __name__ == "__main__":
    unittest.main()