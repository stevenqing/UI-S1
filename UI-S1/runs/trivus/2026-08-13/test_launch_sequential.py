import sys
import tempfile
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from launch_sequential import artifact_manifest, jobs, worker_command


class LaunchSequentialTest(unittest.TestCase):
    def test_exact_sixty_jobs_per_phase(self):
        for phase in ("cheap", "verifier"):
            values = jobs(phase)
            self.assertEqual(len(values), 60)
            self.assertEqual(len(set(values)), 60)

    def test_worker_command_binds_gpu_receipt_and_attempt(self):
        command, environment = worker_command(
            "python", "cheap", (0, 1, "mind2web"),
            Path("receipt.json"), Path("attempt"), 7,
        )
        self.assertIn("--authorization-receipt", command)
        self.assertIn("--output-root", command)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "7")

    def test_manifest_rejects_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                artifact_manifest(Path(directory))


if __name__ == "__main__":
    unittest.main()