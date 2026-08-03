import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUN_DIR = ROOT / "runs/diversity-axis/2026-08-02"
SCRIPT = RUN_DIR / "x2/generate_microchains.py"
INPUTS = RUN_DIR / "raw/x2_inputs.jsonl"


JOBS = (
    (0, "Q2", "gta1", 0),
    (1, "Q2", "gta1", 1),
    (2, "Q4", "gta1", 0),
    (3, "Q4", "gta1", 1),
    (4, "Q4", "qwen3", 0),
    (5, "Q4", "qwen3", 1),
    (6, "Q4", "uitars", 0),
    (7, "Q4", "uitars", 1),
)


def command(cell, model_type, shard):
    python = ROOT / (".venv-ac-vllm/bin/python" if model_type == "qwen3" else "runs/mind2web-tongui/2026-07-28/.venv/bin/python")
    return [
        str(python), str(SCRIPT), "--inputs", str(INPUTS), "--cell", cell,
        "--model-type", model_type, "--num-shards", "2", "--shard-index", str(shard),
        "--output", str(RUN_DIR / f"shards/{cell.lower()}-{model_type}-{shard}.jsonl"), "--resume",
    ]


def environment(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(RUN_DIR / "x2")
    return env


def main():
    if not INPUTS.exists():
        raise FileNotFoundError(INPUTS)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "shards").mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu, cell, model_type, shard in JOBS:
        log_path = RUN_DIR / f"logs/{cell.lower()}-{model_type}-{shard}.log"
        log = log_path.open("a")
        process = subprocess.Popen(command(cell, model_type, shard), cwd=ROOT, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
        processes.append((cell, model_type, shard, process, log, log_path))
        print(f"started {cell} {model_type} shard {shard} on GPU {gpu}: PID {process.pid}", flush=True)
    failures = []
    try:
        for cell, model_type, shard, process, log, log_path in processes:
            returncode = process.wait()
            log.close()
            print(f"finished {cell} {model_type} shard {shard}: exit {returncode}", flush=True)
            if returncode:
                failures.append((cell, model_type, shard, returncode, log_path))
    except BaseException:
        for _, _, _, process, log, _ in processes:
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