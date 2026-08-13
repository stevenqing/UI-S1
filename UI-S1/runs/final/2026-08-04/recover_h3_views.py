import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
REGIONS = RUN_DIR / "raw/shared_regions_n4.jsonl"
SCRIPT = RUN_DIR / "generate_fixed_regions.py"
PROTECTED_PID = int(os.environ.get("PROTECTED_PID", "2274"))


def protected_alive():
    return Path(f"/proc/{PROTECTED_PID}").is_dir()


def command(model, shard):
    common = [
        str(SCRIPT), "--regions", str(REGIONS), "--views", "4",
        "--num-shards", "4", "--shard-index", str(shard), "--resume",
    ]
    if model == "qwen3":
        python = ROOT / ".venv-ac-vllm/bin/python"
        specific = [
            "--model-dir", str(RUN_DIR / "models/Qwen3-VL-8B-Instruct"),
            "--model-type", "qwen3", "--model-id", "Qwen3-VL-8B-Instruct",
            "--model-revision", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            "--output", str(RUN_DIR / f"shards/qwen3_views/shard-{shard}.jsonl"),
        ]
    else:
        python = ROOT / "runs/mind2web-tongui/2026-07-28/.venv/bin/python"
        specific = [
            "--model-dir", str(ROOT / "runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT"),
            "--model-type", "uitars", "--model-id", "UI-TARS-7B-SFT",
            "--model-revision", "3434901a9dd04dd3625617d839a5724fe5e2db20",
            "--output", str(RUN_DIR / f"shards/uitars_views/shard-{shard}.jsonl"),
        ]
    return [str(python), *common, *specific]


def main():
    if not protected_alive():
        raise RuntimeError(f"protected PID {PROTECTED_PID} is not alive before H3")
    if not REGIONS.is_file():
        raise FileNotFoundError(REGIONS)
    for path in (RUN_DIR / "logs", RUN_DIR / "shards/qwen3_views", RUN_DIR / "shards/uitars_views"):
        path.mkdir(parents=True, exist_ok=True)
    workers = []
    for gpu in range(8):
        model = "qwen3" if gpu < 4 else "uitars"
        shard = gpu if gpu < 4 else gpu - 4
        log_path = RUN_DIR / f"logs/recover-{model}-shard-{shard}.log"
        log = log_path.open("a")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if model == "qwen3":
            environment.pop("PYTHONPATH", None)
        else:
            environment["PYTHONPATH"] = os.pathsep.join([
                str(ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"),
                str(ROOT / "runs/collision-law/2026-07-30/w3_assets/mvp-overlay"),
            ])
        process = subprocess.Popen(command(model, shard), cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        workers.append((model, shard, process, log, log_path))
        print(f"started {model} shard {shard} on GPU {gpu}: PID {process.pid}", flush=True)
    failures = []
    try:
        for model, shard, process, log, log_path in workers:
            returncode = process.wait()
            log.close()
            print(f"finished {model} shard {shard}: exit {returncode}", flush=True)
            if returncode:
                failures.append((model, shard, returncode, str(log_path)))
    except BaseException:
        for _, _, process, log, _ in workers:
            if process.poll() is None:
                process.terminate()
            log.close()
        raise
    if failures:
        raise RuntimeError(f"H3 recovery failures: {failures}")
    if not protected_alive():
        raise RuntimeError(f"protected PID {PROTECTED_PID} disappeared during H3")
    print("H3_RECOVERY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
