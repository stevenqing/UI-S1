import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "runs/closing/2026-08-02"
SCRIPT = RUN_DIR / "k8_x2_generate.py"
INPUTS = RUN_DIR / "raw/inputs.jsonl"
JOBS = (
    (0, "single", "gta1", 0, 4),
    (1, "single", "gta1", 1, 4),
    (2, "single", "gta1", 2, 4),
    (3, "single", "gta1", 3, 4),
    (4, "mixed", "gta1", 0, 1),
    (5, "mixed", "qwen3", 0, 1),
    (6, "mixed", "uitars", 0, 2),
    (7, "mixed", "uitars", 1, 2),
)


def command(family, model_type, shard, num_shards):
    python = ROOT / (".venv-ac-vllm/bin/python" if model_type == "qwen3" else "runs/mind2web-tongui/2026-07-28/.venv/bin/python")
    return [
        str(python), str(SCRIPT), "--inputs", str(INPUTS), "--family", family,
        "--model-type", model_type, "--num-shards", str(num_shards),
        "--shard-index", str(shard),
        "--output", str(RUN_DIR / f"shards/k8-{family}-{model_type}-{shard}.jsonl"),
        "--resume",
    ]


def main():
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "shards").mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu, family, model_type, shard, num_shards in JOBS:
        log_path = RUN_DIR / f"logs/k8-{family}-{model_type}-{shard}.log"
        log = log_path.open("a")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(ROOT / "runs/diversity-axis/2026-08-02/x2")
        process = subprocess.Popen(
            command(family, model_type, shard, num_shards), cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((family, model_type, shard, process, log, log_path))
        print(f"started K8 {family} {model_type} shard {shard} on GPU {gpu}: PID {process.pid}", flush=True)
    failures = []
    try:
        for family, model_type, shard, process, log, log_path in processes:
            returncode = process.wait()
            log.close()
            print(f"finished K8 {family} {model_type} shard {shard}: exit {returncode}", flush=True)
            if returncode:
                failures.append((family, model_type, shard, returncode, log_path))
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