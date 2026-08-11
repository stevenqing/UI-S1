import os
import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PYTHON = ROOT / ".venv-scaleup/bin/python"
TRAIN = RUN_DIR / "set_ranker_train.py"


def main():
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 is unexpectedly absent")
    output_dir = RUN_DIR / "set_ranker"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for outer_fold in range(5):
        output = output_dir / f"outer-{outer_fold}.json"
        pretest = output_dir / f"outer-{outer_fold}.pretest.json"
        if output.exists() or pretest.exists():
            raise FileExistsError(output if output.exists() else pretest)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(outer_fold)
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        log_path = log_dir / f"outer-{outer_fold}.log"
        log = log_path.open("w", buffering=1)
        process = subprocess.Popen(
            [str(PYTHON), str(TRAIN), "--outer-fold", str(outer_fold), "--output", str(output)],
            cwd=RUN_DIR, env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((outer_fold, process, log, log_path))
        print(f"started VUS-SR outer={outer_fold} gpu={outer_fold} pid={process.pid}", flush=True)
    failures = []
    try:
        for outer_fold, process, log, log_path in processes:
            returncode = process.wait()
            log.close()
            print(f"finished VUS-SR outer={outer_fold} exit={returncode}", flush=True)
            if returncode:
                failures.append((outer_fold, returncode, str(log_path)))
    except BaseException:
        for _, process, log, _ in processes:
            if process.poll() is None:
                process.terminate()
            log.close()
        raise
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 disappeared during VUS-SR")
    if failures:
        print(f"VUS-SR failures: {failures}", file=sys.stderr)
        raise SystemExit(1)
    print("VUS_SET_RANKER_ALL_OUTERS_PASS", flush=True)


if __name__ == "__main__":
    main()
