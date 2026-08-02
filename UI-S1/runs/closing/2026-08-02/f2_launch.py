import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "runs/closing/2026-08-02"
SCRIPT = RUN_DIR / "f2_generate_samples.py"
INPUTS = RUN_DIR / "raw/inputs.jsonl"
PYTHON = ROOT / "runs/mind2web-tongui/2026-07-28/.venv/bin/python"


def command(shard):
    return [
        str(PYTHON), str(SCRIPT), "--inputs", str(INPUTS),
        "--num-shards", "8", "--shard-index", str(shard),
        "--output", str(RUN_DIR / f"shards/f2-samples-{shard}.jsonl"),
        "--resume",
    ]


def main():
    if not INPUTS.exists():
        raise FileNotFoundError(INPUTS)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "shards").mkdir(parents=True, exist_ok=True)
    processes = []
    for shard in range(8):
        log_path = RUN_DIR / f"logs/f2-samples-{shard}.log"
        log = log_path.open("a")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        process = subprocess.Popen(command(shard), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((shard, process, log, log_path))
        print(f"started F2 shard {shard} on GPU {shard}: PID {process.pid}", flush=True)
    failures = []
    try:
        for shard, process, log, log_path in processes:
            returncode = process.wait()
            log.close()
            print(f"finished F2 shard {shard}: exit {returncode}", flush=True)
            if returncode:
                failures.append((shard, returncode, log_path))
    except BaseException:
        for _, process, log, _ in processes:
            if process.poll() is None:
                process.terminate()
            log.close()
        raise
    if failures:
        for failure in failures:
            print(f"FAILED: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()