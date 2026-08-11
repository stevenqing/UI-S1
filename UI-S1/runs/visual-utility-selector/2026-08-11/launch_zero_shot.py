import argparse
import os
import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PYTHON = ROOT / ".venv-ac-vllm/bin/python"
INFER = RUN_DIR / "zero_shot_infer.py"


def command(shard, args):
    values = [
        str(PYTHON), str(INFER),
        "--records", str(args.records),
        "--model-dir", str(args.model_dir),
        "--output", str(args.output_dir / f"shard-{shard}.jsonl"),
        "--num-shards", "8",
        "--shard-index", str(shard),
        "--batch-size", str(args.batch_size),
        "--max-edge", str(args.max_edge),
        "--resume",
    ]
    if args.limit is not None:
        values.extend(("--limit", str(args.limit)))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=RUN_DIR / "data/public_records.jsonl")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR / "zero_shot/raw")
    parser.add_argument("--log-dir", type=Path, default=RUN_DIR / "zero_shot/logs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 is unexpectedly absent")
    if not args.records.is_file():
        raise FileNotFoundError("VUS public records are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu in range(8):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["TOKENIZERS_PARALLELISM"] = "false"
        log_path = args.log_dir / f"shard-{gpu}.log"
        log = log_path.open("a", buffering=1)
        process = subprocess.Popen(
            command(gpu, args), cwd=RUN_DIR, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((gpu, process, log, log_path))
        print(f"started VUS shard={gpu} gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    failures = []
    try:
        for gpu, process, log, log_path in processes:
            returncode = process.wait()
            log.close()
            print(f"finished VUS shard={gpu} exit={returncode}", flush=True)
            if returncode:
                failures.append((gpu, returncode, str(log_path)))
    except BaseException:
        for _, process, log, _ in processes:
            if process.poll() is None:
                process.terminate()
            log.close()
        raise
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 disappeared during VUS inference")
    if failures:
        print(f"VUS failures: {failures}", file=sys.stderr)
        raise SystemExit(1)
    print("VUS_ZERO_SHOT_ALL_SHARDS_PASS", flush=True)


if __name__ == "__main__":
    main()
