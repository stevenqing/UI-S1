import json
import os
import subprocess
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SCALEUP_PYTHON = ROOT / ".venv-scaleup/bin/python"
ANALYSIS_PYTHON = ROOT / ".venv-ac-vllm/bin/python"
INPUTS = ROOT / "runs/closing/2026-08-02/raw/inputs.jsonl"
LABELS = ROOT / "runs/ccm-h2h/2026-07-31/h3/raw/shared_regions_n4.jsonl"
MODELS = (
    ("GTA1-72B", "gta1"),
    ("UI-Venus-Ground-72B", "venus"),
    ("Qwen3.5-122B-A10B", "qwen35"),
)


def validate_trace(path, model):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(rows) != 1581 or len(set(ids)) != 1581:
        raise ValueError(f"G1 incomplete trace: {model}, rows={len(rows)}, unique={len(set(ids))}")
    if any(row["model_id"] != model or "bbox" in row or "target_bbox" in row for row in rows):
        raise ValueError(f"G1 model/target isolation mismatch: {model}")


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        subprocess.run(command, cwd=ROOT, env={**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    raw = RUN_DIR / "raw"
    logs = RUN_DIR / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    for model, slug in MODELS:
        output = raw / f"g1-{slug}.jsonl"
        if output.exists():
            try:
                validate_trace(output, model)
                continue
            except ValueError:
                pass
        run_logged([
            str(SCALEUP_PYTHON), str(RUN_DIR / "g1_generate.py"),
            "--inputs", str(INPUTS), "--model", model,
            "--model-dir", str(RUN_DIR / "models" / model),
            "--output", str(output), "--batch-size", "8",
            "--gpu-memory-utilization", "0.68", "--resume",
        ], logs / f"g1-{slug}.log")
        validate_trace(output, model)
    subprocess.run([
        str(ANALYSIS_PYTHON), str(RUN_DIR / "g1_lineage_gate.py"),
        "--venus", str(raw / "g1-venus.jsonl"),
        "--gta1", str(raw / "g1-gta1.jsonl"),
        "--qwen35", str(raw / "g1-qwen35.jsonl"),
        "--labels", str(LABELS), "--output", str(RUN_DIR / "g1_lineage_gate.json"),
    ], cwd=ROOT, check=True)
    result = json.loads((RUN_DIR / "g1_lineage_gate.json").read_text())
    print(json.dumps({"status": result["status"], "pass_at_3": result["pass_at_3"], "gate": result["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()