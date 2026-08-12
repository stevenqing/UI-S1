import os
import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
PYTHON = Path(sys.executable).resolve()


def main():
    output_dir = RUN_DIR / "outer"
    logs = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for fold in range(5):
        output = output_dir / f"outer-{fold}.json"
        pretest = output_dir / f"outer-{fold}.pretest.json"
        if output.exists() or pretest.exists():
            raise FileExistsError(output)
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "4"
        environment["MKL_NUM_THREADS"] = "4"
        environment["OPENBLAS_NUM_THREADS"] = "4"
        log_path = logs / f"outer-{fold}.log"
        log = log_path.open("w", buffering=1)
        process = subprocess.Popen([
            str(PYTHON), str(RUN_DIR / "civa_train.py"),
            "--outer-fold", str(fold), "--output", str(output),
        ], cwd=RUN_DIR, env=environment, stdout=log, stderr=subprocess.STDOUT)
        processes.append((fold, process, log, log_path))
        print(f"started CIVA outer={fold} pid={process.pid}", flush=True)
    failures = []
    for fold, process, log, path in processes:
        code = process.wait()
        log.close()
        print(f"finished CIVA outer={fold} exit={code}", flush=True)
        if code:
            failures.append((fold, code, str(path)))
    if failures:
        print(failures, file=sys.stderr)
        raise SystemExit(1)
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 disappeared")
    print("CIVA_ALL_OUTERS_PASS", flush=True)


if __name__ == "__main__":
    main()