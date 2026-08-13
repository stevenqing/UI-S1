import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch_trivus
import formal_authorization


class LaunchTriVUSTest(unittest.TestCase):
    def test_missing_authorization_fails_closed(self):
        original = formal_authorization.AUTHORIZATION_PATH
        formal_authorization.AUTHORIZATION_PATH = Path(__file__).parent / "missing-formal-authorization.json"
        try:
            with self.assertRaises(PermissionError):
                launch_trivus.validate_new_authorization()
        finally:
            formal_authorization.AUTHORIZATION_PATH = original

    @staticmethod
    def complete_outer(attempt, fold):
        outer = attempt / f"outer-{fold}"
        outer.mkdir(parents=True)
        pretest = outer / f"outer-{fold}.pretest.json"
        pretest.write_text(json.dumps({"fold": fold}))
        result = outer / f"outer-{fold}.json"
        result_value = {
            "schema_version": 1,
            "status": "PASS_TRIVUS_OUTER_COMPLETE",
            "outer_fold": fold,
            "pretest_sha256": launch_trivus.sha256_file(pretest),
            "outputs": {policy: {} for policy in launch_trivus.POLICY_SPECS},
        }
        result.write_text(json.dumps(result_value))
        (outer / "OUTER_COMPLETE.json").write_text(json.dumps({
            "schema_version": 1,
            "status": "TRIVUS_OUTER_COMPLETE",
            "outer_fold": fold,
            "result_sha256": launch_trivus.sha256_file(result),
        }))

    def test_publish_attempt_requires_all_five_complete_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            output = root / "formal"
            for fold in range(5):
                self.complete_outer(attempt, fold)
            original = launch_trivus.OUTPUT_ROOT
            launch_trivus.OUTPUT_ROOT = output
            try:
                launch_trivus.publish_attempt(attempt)
                self.assertTrue(all((output / f"outer-{fold}").is_dir() for fold in range(5)))
                self.assertFalse(attempt.exists())
            finally:
                launch_trivus.OUTPUT_ROOT = original

    def test_publish_attempt_rejects_missing_marker_without_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            output = root / "formal"
            for fold in range(5):
                (attempt / f"outer-{fold}").mkdir(parents=True)
            original = launch_trivus.OUTPUT_ROOT
            launch_trivus.OUTPUT_ROOT = output
            try:
                with self.assertRaises(RuntimeError):
                    launch_trivus.publish_attempt(attempt)
                self.assertFalse(output.exists())
            finally:
                launch_trivus.OUTPUT_ROOT = original

    def test_publish_attempt_rejects_result_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            output = root / "formal"
            for fold in range(5):
                self.complete_outer(attempt, fold)
            (attempt / "outer-2" / "outer-2.json").write_text("{}")
            original = launch_trivus.OUTPUT_ROOT
            launch_trivus.OUTPUT_ROOT = output
            try:
                with self.assertRaises(RuntimeError):
                    launch_trivus.publish_attempt(attempt)
                self.assertFalse(output.exists())
            finally:
                launch_trivus.OUTPUT_ROOT = original

    def test_worker_command_preserves_receipt_and_uses_local_cuda_zero(self):
        receipt = Path("/tmp/receipt.json")
        for fold in range(5):
            command, environment = launch_trivus.worker_command(
                "/python", fold, receipt, {"BASE": "1"}
            )
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], str(fold))
            self.assertEqual(command[command.index("--device") + 1], "cuda:0")
            self.assertEqual(
                command[command.index("--authorization-receipt") + 1],
                str(receipt),
            )


if __name__ == "__main__":
    unittest.main()