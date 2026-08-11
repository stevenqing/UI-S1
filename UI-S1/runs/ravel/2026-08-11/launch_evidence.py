import argparse
import os
import subprocess
import sys
from pathlib import Path

from evidence_data import MODES


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PYTHON = ROOT / ".venv-ac-vllm/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 absent")
    raw = RUN_DIR / f"evidence/{args.mode}/raw"
    logs = RUN_DIR / f"evidence/{args.mode}/logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu in range(8):
        output = raw / f"shard-{gpu}.jsonl"
        command = [
            str(PYTHON), str(RUN_DIR / "evidence_infer.py"),
            "--mode", args.mode, "--output", str(output),
            "--num-shards", "8", "--shard-index", str(gpu), "--resume",
        ]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["TOKENIZERS_PARALLELISM"] = "false"
        log_path = logs / f"shard-{gpu}.log"
        log = log_path.open("a", buffering=1)
        process = subprocess.Popen(command, cwd=RUN_DIR, env=environment, stdout=log, stderr=subprocess.STDOUT)
        processes.append((gpu, process, log, log_path))
        print(f"started RAVEL mode={args.mode} shard={gpu} gpu={gpu} pid={process.pid}", flush=True)
    failures = []
    for gpu, process, log, path in processes:
        code = process.wait()
        log.close()
        print(f"finished RAVEL mode={args.mode} shard={gpu} exit={code}", flush=True)
        if code:
            failures.append((gpu, code, str(path)))
    if failures:
        print(failures, file=sys.stderr)
        raise SystemExit(1)
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 disappeared")
    print(f"RAVEL_{args.mode.upper()}_ALL_SHARDS_PASS", flush=True)


if __name__ == "__main__":
    main()
