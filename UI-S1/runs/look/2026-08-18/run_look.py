import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
FORMAL_PATH = RUN_DIR / "INFERENCE_INPUT_MANIFEST.jsonl"
SMOKE_PATH = RUN_DIR / "SMOKE_INPUT_MANIFEST.jsonl"
AUTHORIZATION_PATH = RUN_DIR / "EXECUTION_AUTHORIZATION.json"
SMOKE_STATUS_PATH = RUN_DIR / "SMOKE_STATUS.json"
NONCE_PATH = RUN_DIR / "raw/NONCE_CONSUMED.json"
OWIN_RUNNER_PATH = ROOT / "runs/owin/2026-08-17/run_arm_a.py"
FORBIDDEN = {"target_bbox", "bbox", "correct", "correctness", "mode_correctness", "random_correctness", "stratum", "label", "reward"}


def load_owin_runner():
    spec = importlib.util.spec_from_file_location("look_owin_inference_kernel", OWIN_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recursive_keys(value):
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(recursive_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(recursive_keys(child))
        return result
    return set()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_rows(rows, expected):
    if len(rows) != expected or len({row["row_id"] for row in rows}) != expected:
        raise ValueError("LOOK input row mismatch")
    if recursive_keys(rows) & FORBIDDEN:
        raise ValueError("LOOK input contains evaluation fields")
    if any(len(row["windows"]) != 3 or [window["kind"] for window in row["windows"]] != ["main", "sensitivity", "null"] for row in rows):
        raise ValueError("LOOK input windows mismatch")


def authorization():
    if not AUTHORIZATION_PATH.exists():
        raise PermissionError("LOOK GPU execution is not authorized")
    value = json.loads(AUTHORIZATION_PATH.read_text())
    if value.get("status") != "AUTHORIZED_ONE_TIME_LOOK" or value.get("formal_calls") != 1290:
        raise PermissionError("LOOK authorization mismatch")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate-only", "smoke", "formal"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    formal = read_jsonl(FORMAL_PATH)
    smoke = read_jsonl(SMOKE_PATH)
    validate_rows(formal, 430)
    validate_rows(smoke, 3)
    if args.mode == "validate-only":
        print(json.dumps({"status": "PASS_LOOK_RUNNER_VALIDATION", "formal_rows": 430, "formal_calls": 1290, "smoke_rows": 3, "gpu_used": False, "gpu_authorized": AUTHORIZATION_PATH.exists()}, indent=2))
        return
    auth = authorization()
    if args.mode == "formal" and (not SMOKE_STATUS_PATH.exists() or json.loads(SMOKE_STATUS_PATH.read_text()).get("status") != "PASS_LOOK_SMOKE_9"):
        raise PermissionError("LOOK formal run requires passing smoke")
    rows = smoke if args.mode == "smoke" else formal
    output = RUN_DIR / ("smoke/traces.jsonl" if args.mode == "smoke" else "raw/formal_traces.jsonl")
    if output.exists():
        raise FileExistsError(output)
    if args.mode == "formal" and not NONCE_PATH.exists():
        atomic_json(NONCE_PATH, {"status": "CONSUMED", "authorization_sha256": sha256_file(AUTHORIZATION_PATH)})
    owin = load_owin_runner()
    torch, mvp, model, processor = owin.load_runtime(args.device)
    backend = {"name": "historical_H1_transformers_overlay", "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__, "device": args.device}
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            for slot, window in enumerate(row["windows"]):
                runner_window = {"slot": slot, **window}
                try:
                    trace = owin.run_window(row, runner_window, torch, mvp, model, processor, args.device, backend)
                except Exception as error:
                    failures += 1
                    trace = {"schema_version": 1, "status": "failed", "sample_id": row["sample_id"], "row_id": row["row_id"], "slot": slot, "window": runner_window, "backend": backend, "error_type": type(error).__name__, "error": str(error)}
                if recursive_keys(trace) & FORBIDDEN:
                    raise ValueError("LOOK trace contains evaluation fields")
                handle.write(json.dumps(trace, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    calls = len(rows) * 3
    status = {"status": "PASS_LOOK_SMOKE_9" if args.mode == "smoke" and failures == 0 else "LOOK_FORMAL_COMPLETE", "mode": args.mode, "rows": len(rows), "calls": calls, "failures": failures, "failure_rate": failures / calls, "trace_path": str(output.relative_to(ROOT)), "trace_bytes": output.stat().st_size, "trace_sha256": sha256_file(output), "authorization_sha256": sha256_file(AUTHORIZATION_PATH)}
    status_path = SMOKE_STATUS_PATH if args.mode == "smoke" else RUN_DIR / "raw/formal_status.json"
    atomic_json(status_path, status)
    print(json.dumps(status, indent=2))
    if failures / calls > 0.01 or (args.mode == "smoke" and failures):
        raise RuntimeError("LOOK failure threshold exceeded")


if __name__ == "__main__":
    main()