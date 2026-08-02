import json
import os
import subprocess
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ANALYSIS_PYTHON = ROOT / ".venv-ac-vllm/bin/python"
SCALEUP_PYTHON = ROOT / ".venv-scaleup/bin/python"
LEGACY_PYTHON = ROOT / "runs/mind2web-tongui/2026-07-28/.venv/bin/python"
INPUTS = ROOT / "runs/closing/2026-08-02/raw/inputs.jsonl"
LABELS = ROOT / "runs/ccm-h2h/2026-07-31/h3/raw/shared_regions_n4.jsonl"
MODELS = (
    ("GTA1-72B", "gta1"),
    ("UI-Venus-Ground-72B", "venus"),
    ("Qwen3.5-122B-A10B", "qwen35"),
)


def run_logged(command, name, scaleup_environment=False):
    log_path = RUN_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if scaleup_environment:
        environment.update({"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
    with log_path.open("a") as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def validate_rows(path, source, target_free=True):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(rows) != 1581 or len(set(ids)) != 1581:
        raise ValueError(f"Scale-Up incomplete {source}: rows={len(rows)}, unique={len(set(ids))}")
    if target_free and any("bbox" in row or "target_bbox" in row for row in rows):
        raise ValueError(f"Scale-Up target leak: {source}")
    return rows


def score_command(model, slug, regions, output, limit=None):
    command = [
        str(SCALEUP_PYTHON), str(RUN_DIR / "g2_score_regions.py"),
        "--regions", str(regions), "--model", model,
        "--model-dir", str(RUN_DIR / "models" / model),
        "--output", str(output), "--batch-size", "8",
        "--gpu-memory-utilization", "0.68", "--resume",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def main():
    run_logged([str(ANALYSIS_PYTHON), str(RUN_DIR / "g1_run_all.py")], "scaleup-g1")
    g1_path = RUN_DIR / "g1_lineage_gate.json"
    g1 = json.loads(g1_path.read_text())
    if g1["gate"]["G2_cancelled"]:
        run_logged([str(ANALYSIS_PYTHON), str(RUN_DIR / "finalize.py")], "scaleup-finalize")
        print(json.dumps({"status": "COMPLETE", "endpoint": "B_G1_COMMON_FAILURE_CEILING", "G1": g1["gate"]}, indent=2, sort_keys=True))
        return

    raw = RUN_DIR / "raw"
    smoke = RUN_DIR / "smoke"
    raw.mkdir(parents=True, exist_ok=True)
    smoke.mkdir(parents=True, exist_ok=True)
    smoke_regions = smoke / "g2-regions.jsonl"
    run_logged([
        str(LEGACY_PYTHON), str(RUN_DIR / "g2_prepare_regions.py"),
        "--inputs", str(INPUTS), "--model-dir", str(RUN_DIR / "models/GTA1-72B"),
        "--output", str(smoke_regions), "--limit", "1", "--resume",
    ], "g2-regions-smoke")
    smoke_rows = [json.loads(line) for line in smoke_regions.read_text().splitlines() if line.strip()]
    if len(smoke_rows) != 1 or len(smoke_rows[0]["regions"]) < 12:
        raise ValueError("G2 proposer smoke failed")

    regions = raw / "g2-regions.jsonl"
    run_logged([
        str(LEGACY_PYTHON), str(RUN_DIR / "g2_prepare_regions.py"),
        "--inputs", str(INPUTS), "--model-dir", str(RUN_DIR / "models/GTA1-72B"),
        "--output", str(regions), "--resume",
    ], "g2-regions")
    validate_rows(regions, "G2 regions")

    score_paths = {}
    for model, slug in MODELS:
        smoke_output = smoke / f"g2-score-{slug}.jsonl"
        run_logged(score_command(model, slug, regions, smoke_output, limit=1), f"g2-score-{slug}-smoke", scaleup_environment=True)
        smoke_scores = [json.loads(line) for line in smoke_output.read_text().splitlines() if line.strip()]
        if len(smoke_scores) != 1 or not smoke_scores[0]["predictions"]:
            raise ValueError(f"G2 score smoke failed: {model}")
        output = raw / f"g2-score-{slug}.jsonl"
        run_logged(score_command(model, slug, regions, output), f"g2-score-{slug}", scaleup_environment=True)
        validate_rows(output, f"G2 scores {model}")
        score_paths[model] = output

    run_logged([
        str(ANALYSIS_PYTHON), str(RUN_DIR / "g2_mixed_72b.py"),
        "--g1", str(g1_path), "--regions", str(regions),
        "--gta1", str(score_paths["GTA1-72B"]),
        "--venus", str(score_paths["UI-Venus-Ground-72B"]),
        "--qwen35", str(score_paths["Qwen3.5-122B-A10B"]),
        "--labels", str(LABELS), "--output", str(RUN_DIR / "g2_mixed_72b.json"),
    ], "scaleup-g2-analysis")
    run_logged([str(ANALYSIS_PYTHON), str(RUN_DIR / "finalize.py")], "scaleup-finalize")
    g2 = json.loads((RUN_DIR / "g2_mixed_72b.json").read_text())
    print(json.dumps({"status": "COMPLETE", "G1": g1["gate"], "G2": g2["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()