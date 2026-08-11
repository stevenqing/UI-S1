import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(VUS))

from set_ranker_train import run_outer, validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "random", "global_only", "fine_only", "context_only"), required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 absent")
    predictions = RUN_DIR / f"evidence/{args.mode}/predictions.jsonl"
    manifest = RUN_DIR / f"evidence/{args.mode}/predictions.manifest.json"
    if not predictions.is_file() or not manifest.is_file():
        raise FileNotFoundError(predictions)
    manifest_value = json.loads(manifest.read_text())
    if manifest_value["status"] != "PASS_BLIND_EVIDENCE_LOCKED" or manifest_value["private_labels_opened"]:
        raise ValueError("RAVEL visual evidence is not blind-locked")
    config = yaml.safe_load((VUS / "configs/set_ranker_prereg.yaml").read_text())
    validate_config(config)
    pretest = args.output.with_name(f"outer-{args.outer_fold}.pretest.json")
    if args.output.exists() or pretest.exists():
        raise FileExistsError(args.output)
    result = run_outer(
        args.outer_fold, config, torch.device("cuda:0"), pretest,
        predictions_path=predictions,
        predictions_manifest_path=manifest,
    )
    result["ravel_evidence_mode"] = args.mode
    result["ravel_predictions_sha256"] = manifest_value["predictions_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "mode": args.mode,
        "outer_fold": args.outer_fold,
        "selected": result["selected"]["config_id"],
        "final_epochs": result["final_epochs"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
