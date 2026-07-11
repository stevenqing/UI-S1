#!/usr/bin/env python3
"""Export LLaMA-Factory trainer state into the local causal-report schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    state_path = model_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    global_step = int(state.get("global_step") or 0)
    history = list(state.get("log_history") or [])
    loss_rows = []
    for row in history:
        if "loss" not in row:
            continue
        loss_rows.append({
            "epoch": row.get("epoch"),
            "global_step": int(row.get("step") or 0),
            "lm_loss": float(row["loss"]),
            "total_loss": float(row["loss"]),
            "grad_norm": row.get("grad_norm"),
            "learning_rate": row.get("learning_rate"),
            "skipped": 0,
            "kl_loss": 0.0,
            "entropy": None,
            "label_smoothing": 0.0,
            "finetuning_type": "full",
        })
    if not loss_rows:
        train_loss = state.get("train_loss")
        loss_rows.append({
            "epoch": state.get("epoch"),
            "global_step": global_step,
            "lm_loss": None if train_loss is None else float(train_loss),
            "total_loss": None if train_loss is None else float(train_loss),
            "skipped": 0,
            "kl_loss": 0.0,
            "entropy": None,
            "label_smoothing": 0.0,
            "finetuning_type": "full",
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in loss_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    torch.save({"global_step": global_step, "finetuning_type": "full", "model_dir": str(model_dir)}, output_dir / "training_state.pt")
    summary = {
        "global_step": global_step,
        "finetuning_type": "full",
        "model_dir": str(model_dir),
        "loss_rows": len(loss_rows),
        "last_metrics": loss_rows[-1],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
