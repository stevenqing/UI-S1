import os
import subprocess
import sys
from pathlib import Path

from recovery_common import ROOT, RUN_DIR, assert_protected_process, load_config, load_jsonl, references, validate_lane_rows


def build_command(config, lane):
    return [
        str(ROOT / config["python"]),
        str(ROOT / config["source_script"]["path"]),
        "--model-id", lane["model_id"],
        "--setting", lane["setting"],
        "--output", str(ROOT / lane["destination"]),
        "--num-shards", "1",
        "--shard-index", "0",
        "--batch-size", "8",
        "--resume",
    ]


def main():
    config = load_config()
    assert_protected_process(config)
    prepared = RUN_DIR / "recovery/PREPARED.json"
    if not prepared.is_file():
        raise PermissionError("TriVUS R0 seeds must be prepared before launch")
    logs = RUN_DIR / "recovery/logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for name, lane in config["lanes"].items():
        destination = ROOT / lane["destination"]
        current = load_jsonl(destination)
        if len(current) == config["expected_rows_per_lane"]:
            validate_lane_rows(current, references(config, lane["setting"]), lane, require_complete=True)
            print(f"TriVUS R0 lane={name} already complete", flush=True)
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(lane["gpu"])
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        log_path = logs / f"{name}.log"
        log = log_path.open("a", buffering=1)
        process = subprocess.Popen(
            build_command(config, lane), cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((name, lane, process, log, log_path))
        print(f"started TriVUS R0 lane={name} gpu={lane['gpu']} pid={process.pid}", flush=True)
    failures = []
    for name, lane, process, log, path in processes:
        code = process.wait()
        log.close()
        print(f"finished TriVUS R0 lane={name} exit={code}", flush=True)
        if code:
            failures.append((name, code, str(path)))
            continue
        validate_lane_rows(
            load_jsonl(ROOT / lane["destination"]),
            references(config, lane["setting"]), lane, require_complete=True,
        )
    if failures:
        print(failures, file=sys.stderr)
        raise SystemExit(1)
    assert_protected_process(config)
    print("TRIVUS_R0_ALL_RECOVERED_LANES_PASS", flush=True)


if __name__ == "__main__":
    main()