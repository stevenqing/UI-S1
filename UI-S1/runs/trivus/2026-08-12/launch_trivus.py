import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from context_common import fsync_directory, sha256_file
from formal_authorization import (
    OUTPUT_ROOT, consume_authorization, validate_new_authorization,
)
from recovery_common import assert_protected_process, load_config as load_recovery_config
from trivus_outer import POLICY_SPECS


def publish_attempt(attempt):
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    for outer_fold in range(5):
        source = attempt / f"outer-{outer_fold}"
        marker = source / "OUTER_COMPLETE.json"
        result_path = source / f"outer-{outer_fold}.json"
        pretest_path = source / f"outer-{outer_fold}.pretest.json"
        if not marker.is_file() or not result_path.is_file() or not pretest_path.is_file():
            raise RuntimeError(f"TriVUS outer completion marker missing: {outer_fold}")
        marker_value = json.loads(marker.read_text())
        result = json.loads(result_path.read_text())
        if (
            marker_value != {
                "schema_version": 1,
                "status": "TRIVUS_OUTER_COMPLETE",
                "outer_fold": outer_fold,
                "result_sha256": sha256_file(result_path),
            }
            or result.get("schema_version") != 1
            or result.get("status") != "PASS_TRIVUS_OUTER_COMPLETE"
            or result.get("outer_fold") != outer_fold
            or result.get("pretest_sha256") != sha256_file(pretest_path)
            or set(result.get("outputs", {})) != set(POLICY_SPECS)
        ):
            raise RuntimeError(f"TriVUS invalid outer completion marker: {outer_fold}")
    attempt.rename(OUTPUT_ROOT)
    fsync_directory(OUTPUT_ROOT.parent)


def worker_command(python, outer_fold, receipt, environment=None):
    command = [
        python,
        str(RUN_DIR / "trivus_outer.py"),
        "--outer-fold", str(outer_fold),
        "--device", "cuda:0",
        "--authorization-receipt", str(receipt),
    ]
    values = dict(os.environ if environment is None else environment)
    values["CUDA_VISIBLE_DEVICES"] = str(outer_fold)
    return command, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(ROOT / ".venv-scaleup/bin/python"))
    args = parser.parse_args()
    recovery = load_recovery_config()
    assert_protected_process(recovery)
    authorization, authorization_commit, receipt, attempt = validate_new_authorization()
    consume_authorization(authorization, authorization_commit, receipt)
    processes = []
    for outer_fold in range(5):
        command, environment = worker_command(
            args.python, outer_fold, receipt
        )
        process = subprocess.Popen(command, cwd=ROOT, env=environment)
        processes.append((outer_fold, process))
    failures = []
    for outer_fold, process in processes:
        returncode = process.wait()
        if returncode:
            failures.append({"outer_fold": outer_fold, "returncode": returncode})
    assert_protected_process(recovery)
    if failures:
        raise RuntimeError(f"TriVUS outer worker failures: {failures}")
    publish_attempt(attempt)
    print(json.dumps({
        "status": "PASS_TRIVUS_ALL_OUTER_WORKERS_COMPLETE",
        "outer_folds": list(range(5)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()