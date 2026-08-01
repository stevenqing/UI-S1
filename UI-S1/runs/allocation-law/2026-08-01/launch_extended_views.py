import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "runs/allocation-law/2026-08-01"
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
REGIONS = RUN_DIR / "raw/shared_regions_n12.jsonl"
GENERATOR = RUN_DIR / "generate_extended_views.py"


def command(model, shard):
    common = [
        "--regions", str(REGIONS),
        "--view-start", "4", "--view-stop", "12",
        "--num-shards", "4", "--shard-index", str(shard), "--resume",
    ]
    if model == "qwen3":
        python = ROOT / ".venv-ac-vllm/bin/python"
        model_dir = H3_DIR / "models/Qwen3-VL-8B-Instruct"
        specific = [
            "--model-dir", str(model_dir), "--model-type", "qwen3",
            "--model-id", "Qwen3-VL-8B-Instruct",
            "--model-revision", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            "--output", str(RUN_DIR / f"shards/qwen3-views-4-11-{shard}.jsonl"),
        ]
    else:
        python = ROOT / "runs/mind2web-tongui/2026-07-28/.venv/bin/python"
        model_dir = ROOT / "runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT"
        specific = [
            "--model-dir", str(model_dir), "--model-type", "uitars",
            "--model-id", "UI-TARS-7B-SFT",
            "--model-revision", "3434901a9dd04dd3625617d839a5724fe5e2db20",
            "--output", str(RUN_DIR / f"shards/uitars-views-4-11-{shard}.jsonl"),
        ]
    return [str(python), str(GENERATOR), *common, *specific]


def environment(model, gpu):
    env = os.environ.copy()
    paths = [str(RUN_DIR), str(H3_DIR)]
    if model == "uitars":
        paths.extend([
            str(ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"),
            str(ROOT / "runs/collision-law/2026-07-30/w3_assets/mvp-overlay"),
        ])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def main():
    if not REGIONS.exists():
        raise FileNotFoundError(REGIONS)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "shards").mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu in range(8):
        model = "qwen3" if gpu < 4 else "uitars"
        shard = gpu if gpu < 4 else gpu - 4
        log_path = RUN_DIR / f"logs/{model}-views-4-11-{shard}.log"
        log = log_path.open("a")
        process = subprocess.Popen(
            command(model, shard), cwd=ROOT, env=environment(model, gpu),
            stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((model, shard, process, log, log_path))
        print(f"started {model} shard {shard} on GPU {gpu}: PID {process.pid}", flush=True)

    failures = []
    try:
        for model, shard, process, log, log_path in processes:
            returncode = process.wait()
            log.close()
            print(f"finished {model} shard {shard}: exit {returncode}", flush=True)
            if returncode:
                failures.append((model, shard, returncode, log_path))
    except BaseException:
        for _, _, process, log, _ in processes:
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