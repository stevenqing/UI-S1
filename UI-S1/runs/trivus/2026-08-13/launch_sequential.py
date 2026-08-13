import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(RUN_DIR.parent / "2026-08-12"))

from context_common import atomic_json_file, fsync_directory, sha256_file
from recovery_common import assert_protected_process, load_config as load_recovery_config
from sequential_authorization import (
    OUTPUT_ROOT, consume_authorization, validate_new_authorization,
)


FAMILIES = ("mind2web", "screenspot_pro", "androidcontrol")


def jobs(phase):
    return [
        (outer, holdout, family)
        for outer in range(5)
        for holdout in range(5)
        if holdout != outer
        for family in FAMILIES
    ]


def worker_command(python, phase, job, receipt, attempt, gpu):
    outer, holdout, family = job
    script = "sequential_oof_runner.py" if phase == "cheap" else "sequential_verifier_oof_runner.py"
    command = [
        python, str(RUN_DIR / script),
        "--outer-fold", str(outer),
        "--holdout-fold", str(holdout),
        "--family", family,
        "--device", "cuda:0",
        "--authorization-receipt", str(receipt),
        "--output-root", str(attempt / phase),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return command, environment


def run_phase(python, phase, receipt, attempt, gpu_count):
    pending = list(jobs(phase))
    active = []
    failures = []
    log_root = attempt / "logs" / phase
    log_root.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and len(active) < gpu_count:
            job = pending.pop(0)
            gpu = next(index for index in range(gpu_count) if index not in {item[0] for item in active})
            outer, holdout, family = job
            log_path = log_root / f"outer-{outer}-holdout-{holdout}-{family}.log"
            handle = log_path.open("w")
            command, environment = worker_command(
                python, phase, job, receipt, attempt, gpu
            )
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            active.append((gpu, process, job, handle))
        gpu, process, job, handle = active.pop(0)
        returncode = process.wait()
        handle.close()
        if returncode:
            failures.append({"job": job, "returncode": returncode})
            for _, other, _, other_handle in active:
                if other.poll() is None:
                    other.terminate()
            for _, other, _, other_handle in active:
                other.wait()
                other_handle.close()
            active.clear()
            break
    if failures:
        raise RuntimeError(f"Sequential {phase} worker failures: {failures}")


def artifact_manifest(attempt):
    artifacts = {}
    for phase in ("cheap", "verifier"):
        root = attempt / phase
        for job in jobs(phase):
            outer, holdout, family = job
            directory = root / f"outer-{outer}" / f"holdout-{holdout}"
            for suffix in (".pt", ".jsonl"):
                path = directory / f"{family}{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(path)
                relative = path.relative_to(attempt).as_posix()
                artifacts[relative] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
    if len(artifacts) != 240:
        raise RuntimeError("Sequential artifact count mismatch")
    return artifacts


def publish_attempt(attempt, receipt):
    artifacts = artifact_manifest(attempt)
    manifest = {
        "schema_version": 1,
        "status": "PASS_EXPLORATORY_SEQUENTIAL_OOF_COMPLETE",
        "confirmatory": False,
        "promotion_allowed": False,
        "receipt_sha256": sha256_file(receipt),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    atomic_json_file(attempt / "MANIFEST.json", manifest)
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    attempt.rename(OUTPUT_ROOT)
    fsync_directory(OUTPUT_ROOT.parent)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(ROOT / ".venv-scaleup/bin/python"))
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.gpus <= 8:
        raise ValueError("Sequential launcher GPU count must be 1..8")
    recovery = load_recovery_config()
    assert_protected_process(recovery)
    authorization, authorization_commit, receipt, attempt = validate_new_authorization()
    consume_authorization(authorization, authorization_commit, receipt)
    attempt.mkdir(parents=False, exist_ok=False)
    for phase in ("cheap", "verifier"):
        run_phase(args.python, phase, receipt, attempt, args.gpus)
        assert_protected_process(recovery)
    manifest = publish_attempt(attempt, receipt)
    print(json.dumps({
        "status": manifest["status"],
        "artifact_count": manifest["artifact_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()