import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import formal_authorization


class FormalAuthorizationTest(unittest.TestCase):
    def test_missing_committed_authorization_fails_closed(self):
        original = formal_authorization.AUTHORIZATION_PATH
        formal_authorization.AUTHORIZATION_PATH = Path(__file__).parent / "missing-formal-authorization.json"
        try:
            with self.assertRaises(PermissionError):
                formal_authorization.load_bound_authorization()
        finally:
            formal_authorization.AUTHORIZATION_PATH = original

    def test_nonce_paths_are_hidden_and_canonical(self):
        nonce = "a" * 64
        self.assertEqual(
            formal_authorization.attempt_path(nonce).name,
            f".formal-attempt-{nonce}",
        )
        self.assertEqual(
            formal_authorization.receipt_path(nonce).name,
            f"{nonce}.json",
        )
        with self.assertRaises(PermissionError):
            formal_authorization.attempt_path("../escape")


if __name__ == "__main__":
    unittest.main()