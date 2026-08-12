import os
import subprocess
import sys
from pathlib import Path

from selector_data import RUN_DIR, assert_selector_environment, load_config


PYTHON = Path(sys.executable)


def main():
    config = load_config()
    assert_selector_environment(config)
    output_dir = RUN_DIR / "selector/shards"
    logs = RUN_DIR / "selector/logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard in range(config["inference"]["num_shards"]):
        output = output_dir / f"shard-{shard}.jsonl"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(config["inference"]["gpu_mapping"][shard])
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        log_path = logs / f"shard-{shard}.log"
        log = log_path.open("a", buffering=1)
        command = [
            str(PYTHON), str(RUN_DIR / "selector_infer.py"),
            "--output", str(output), "--shard-index", str(shard), "--resume",
        ]
        process = subprocess.Popen(command, cwd=RUN_DIR.parents[2], env=environment, stdout=log, stderr=subprocess.STDOUT)
        processes.append((shard, process, log, log_path))
        print(f"started TriVUS selector shard={shard} gpu={config['inference']['gpu_mapping'][shard]} pid={process.pid}", flush=True)
    failures = []
    for shard, process, log, path in processes:
        code = process.wait()
        log.close()
        print(f"finished TriVUS selector shard={shard} exit={code}", flush=True)
        if code:
            failures.append((shard, code, str(path)))
    if failures:
        print(failures, file=sys.stderr)
        raise SystemExit(1)
    print("TRIVUS_SELECTOR_ALL_SHARDS_PASS", flush=True)


if __name__ == "__main__":
    main()