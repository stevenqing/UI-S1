#!/usr/bin/env python3
"""Probe whether GUI-360 critical steps are separable in model activations.

This script has two phases:
1. Extract pre-decision hidden states from the frozen GUI-360 SFT model.
2. Train probe-only classifiers over layer x token-position activations.

No generated action or reward is used as probe input. Labels are bottom-k held-out
p_i only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_identifiability import (  # noqa: E402
    auc_score,
    balanced_accuracy,
    build_rows as build_surface_rows,
    feature_name_is_allowed,
    fit_logistic,
    logistic_cv,
    per_signal_metrics,
    read_jsonl,
    triage_table,
    write_json,
    write_jsonl,
)
from v13_gui_360.eval_gui360_template import SUPPORTED_ACTIONS, USER_PROMPT_TEMPLATE  # noqa: E402

DEFAULT_MODEL = "outputs/critstep_verifier_v2/gui360_fullparam_sft_step250_trainview"
DEFAULT_CANDIDATES = "outputs/critstep_binlift_lean/test_candidates/per_step.jsonl"
DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_CRIT_TASKS = "outputs/critstep_eval/per_task.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_representation"
DEFAULT_LAYERS = "8,16,24,28"
DEFAULT_POSITIONS = "prompt_last,vision_mean,vision_max,text_mean"
BASELINE_ENTROPY_AUC = 0.58
BASELINE_SURFACE_AUC = 0.63
BASELINE_INTERNAL_UNCERTAINTY_AUC = 0.634
BASELINE_SURFACE_TOP20_RECALL = 0.335
BUDGETS = (0.10, 0.20, 0.30)


def write_jsonl_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_csv_ints(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_strs(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def read_tasks(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def history_text(history: Any) -> str:
    if isinstance(history, list) and history:
        return "\n".join(str(item) for item in history)
    return "None"


def prompt_text(row: Mapping[str, Any]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        instruction=str(row.get("instruction") or row.get("goal") or ""),
        history=history_text(row.get("history")),
        actions=SUPPORTED_ACTIONS,
    )


def chat_text(processor: Any, row: Mapping[str, Any]) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text(row)}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def tensor_to_device(value: Any, device: torch.device) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value


def layer_to_hidden_index(layer: int, n_layers: int) -> int:
    if layer <= 0:
        return 0
    if layer >= n_layers:
        return n_layers
    return layer


def valid_positions(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], image_token_id: int) -> Dict[str, torch.Tensor]:
    ids = input_ids[0]
    if attention_mask is not None:
        valid = attention_mask[0].bool()
    else:
        valid = torch.ones_like(ids, dtype=torch.bool)
    image = (ids == image_token_id) & valid
    text = valid & (~image)
    return {"valid": valid, "image": image, "text": text}


def masked_pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str, fallback_index: int) -> torch.Tensor:
    if mask.any():
        selected = hidden[mask]
        if mode == "mean":
            return selected.mean(dim=0)
        if mode == "max":
            return selected.max(dim=0).values
        raise ValueError(mode)
    return hidden[fallback_index]


def extract_positions(hidden: torch.Tensor, masks: Mapping[str, torch.Tensor], positions: Sequence[str]) -> Dict[str, torch.Tensor]:
    valid_indices = torch.where(masks["valid"])[0]
    last_idx = int(valid_indices[-1].item()) if len(valid_indices) else hidden.shape[0] - 1
    out: Dict[str, torch.Tensor] = {}
    for position in positions:
        if position == "prompt_last":
            out[position] = hidden[last_idx]
        elif position == "vision_mean":
            out[position] = masked_pool(hidden, masks["image"], "mean", last_idx)
        elif position == "vision_max":
            out[position] = masked_pool(hidden, masks["image"], "max", last_idx)
        elif position == "text_mean":
            out[position] = masked_pool(hidden, masks["text"], "mean", last_idx)
        else:
            raise ValueError(f"unknown position {position}")
    return out


def label_for(row: Mapping[str, Any], task: Mapping[str, Any]) -> Dict[str, Any]:
    step_idx = int(row.get("step_idx") or 0)
    per_p = task.get("per_step_p_heldout_cv") if isinstance(task.get("per_step_p_heldout_cv"), list) else []
    p_i = float(per_p[step_idx]) if step_idx < len(per_p) else float("nan")
    bottom1 = {int(idx) for idx in task.get("bottom1_critical_indices", [])}
    bottom2 = {int(idx) for idx in task.get("bottom2_critical_indices", [])}
    return {
        "bottom1": step_idx in bottom1,
        "bottom2": step_idx in bottom2,
        "p_i_heldout_label_only": p_i,
        "step_log_failure": -math.log(max(1e-8, min(1.0, p_i))) if math.isfinite(p_i) else None,
    }


def extract_shard(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    act_dir = output_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_csv_ints(args.layers)
    positions = parse_csv_strs(args.positions)
    shard_npz = act_dir / f"activations.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz"
    shard_meta = act_dir / f"meta.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.jsonl"
    if shard_npz.exists() and shard_meta.exists() and not args.force:
        print(json.dumps({"skip_existing": str(shard_npz)}, indent=2), flush=True)
        return

    rows_all = read_jsonl(Path(args.candidates))
    if args.limit_steps > 0:
        rows_all = rows_all[: args.limit_steps]
    rows = [(idx, row) for idx, row in enumerate(rows_all) if idx % args.num_shards == args.shard_index]
    tasks = read_tasks(Path(args.crit_tasks))
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    n_layers = int(getattr(config, "num_hidden_layers", max(layers)))
    hidden_size = int(getattr(config, "hidden_size", 0))
    image_token_id = int(getattr(config, "image_token_id", 151655))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(json.dumps({
        "phase": "extract_shard_start",
        "shard": args.shard_index,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "layers": layers,
        "positions": positions,
        "hidden_size": hidden_size,
        "device": str(device),
    }, indent=2), flush=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    arrays: Dict[str, List[np.ndarray]] = {f"L{layer}_{position}": [] for layer in layers for position in positions}
    meta_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.time()
    for local_idx, (global_idx, row) in enumerate(rows, 1):
        try:
            text = chat_text(processor, row)
            image = load_image(str(row["screenshot"]))
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            input_ids = inputs["input_ids"].detach().cpu()
            attention_mask = inputs.get("attention_mask")
            attention_cpu = attention_mask.detach().cpu() if attention_mask is not None else None
            masks = valid_positions(input_ids, attention_cpu, image_token_id)
            inputs = {key: tensor_to_device(value, device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            last_hidden_index = len(hidden_states) - 1
            for layer in layers:
                hidden_index = min(layer_to_hidden_index(layer, n_layers), last_hidden_index)
                hidden = hidden_states[hidden_index][0].detach().float().cpu()
                vectors = extract_positions(hidden, masks, positions)
                for position, vector in vectors.items():
                    arrays[f"L{layer}_{position}"].append(vector.numpy().astype(np.float16))
            task = tasks.get(str(row.get("episode_id")), {})
            label = label_for(row, task)
            meta_rows.append({
                "row_index": global_idx,
                "target_id": row.get("target_id"),
                "episode_id": str(row.get("episode_id")),
                "step_idx": int(row.get("step_idx") or 0),
                "task_k": int(task.get("k") or 0),
                **label,
                "prompt_token_count": int(input_ids.shape[1]),
                "image_token_count": int(masks["image"].sum().item()),
                "text_token_count": int(masks["text"].sum().item()),
                "shard_index": args.shard_index,
            })
            del outputs, hidden_states, inputs
            if device.type == "cuda" and local_idx % 20 == 0:
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            failures.append({"row_index": global_idx, "target_id": row.get("target_id"), "error": str(exc)[:500]})
        if local_idx % args.progress_every == 0 or local_idx == len(rows):
            print(json.dumps({"shard": args.shard_index, "done": local_idx, "total": len(rows), "failures": len(failures), "elapsed_sec": round(time.time() - started, 1)}, ensure_ascii=False), flush=True)
    if failures:
        fail_path = act_dir / f"failures.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.jsonl"
        write_jsonl_rows(fail_path, failures)
    if not meta_rows:
        raise RuntimeError(f"no rows extracted for shard {args.shard_index}")
    np_arrays = {name: np.stack(values, axis=0) for name, values in arrays.items()}
    np.savez(shard_npz, **np_arrays)
    write_jsonl_rows(shard_meta, meta_rows)
    manifest = {
        "model": args.model,
        "candidates": args.candidates,
        "crit_tasks": args.crit_tasks,
        "layers": layers,
        "positions": positions,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "rows": len(meta_rows),
        "failures": len(failures),
        "pre_decision_audit": "Forward pass over prompt with add_generation_prompt=True; no generated action/reward is encoded in the activation input.",
    }
    write_json(act_dir / f"manifest.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json", manifest)
    print(json.dumps({"phase": "extract_shard_done", **manifest, "npz": str(shard_npz)}, indent=2), flush=True)


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for label in (0, 1):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for pos, row_idx in enumerate(idx):
            folds[pos % n_folds].append(int(row_idx))
    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_test = np.where(np.isfinite(x_test), x_test, med)
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (x_train - mean) / scale, (x_test - mean) / scale


def random_project(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if dim <= 0 or dim >= x.shape[1]:
        return x.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(x.shape[1], dim)).astype(np.float32)
    return x.astype(np.float32) @ proj


def logistic_cv_array(x: np.ndarray, y: np.ndarray, n_folds: int, seed: int, l2: float = 1.0) -> Dict[str, Any]:
    folds = stratified_folds(y, n_folds, seed)
    scores = np.zeros(len(y), dtype=float)
    for test_idx in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        x_train, x_test = standardize_train_test(x[train_mask], x[test_idx])
        params = fit_logistic(x_train, y[train_mask], l2=l2)
        logits = np.clip(params[0] + x_test @ params[1:], -40.0, 40.0)
        scores[test_idx] = 1.0 / (1.0 + np.exp(-logits))
    pred = (scores >= 0.5).astype(int)
    return {"scores": scores, "auc": auc_score(y.tolist(), scores.tolist()), "balanced_accuracy": balanced_accuracy(y, pred)}


def mlp_cv_array(x: np.ndarray, y: np.ndarray, n_folds: int, seed: int, epochs: int, hidden_dim: int) -> Dict[str, Any]:
    folds = stratified_folds(y, n_folds, seed)
    scores = np.zeros(len(y), dtype=float)
    torch.manual_seed(seed)
    for fold_idx, test_idx in enumerate(folds):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        x_train, x_test = standardize_train_test(x[train_mask], x[test_idx])
        y_train = y[train_mask].astype(np.float32)
        model = torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(hidden_dim, 1),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        tx = torch.tensor(x_train, dtype=torch.float32)
        ty = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        pos = max(1.0, float(np.sum(y_train == 1)))
        neg = max(1.0, float(np.sum(y_train == 0)))
        weights = torch.tensor(np.where(y_train == 1, 0.5 / pos, 0.5 / neg).reshape(-1, 1), dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logits = model(tx)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty, weight=weights, reduction="sum")
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32)).squeeze(1).numpy()
        scores[test_idx] = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    pred = (scores >= 0.5).astype(int)
    return {"scores": scores, "auc": auc_score(y.tolist(), scores.tolist()), "balanced_accuracy": balanced_accuracy(y, pred)}


def load_activation_shards(output_dir: Path, expected_shards: int) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    act_dir = output_dir / "activations"
    meta_rows: List[Dict[str, Any]] = []
    arrays_by_name: Dict[str, List[np.ndarray]] = defaultdict(list)
    shard_paths = sorted(act_dir.glob("activations.shard_*_of_*.npz"))
    if expected_shards > 0 and len(shard_paths) < expected_shards:
        raise FileNotFoundError(f"expected {expected_shards} activation shards, found {len(shard_paths)}")
    for npz_path in shard_paths:
        suffix = npz_path.name.replace("activations", "meta").replace(".npz", ".jsonl")
        meta_path = act_dir / suffix
        if not meta_path.exists():
            raise FileNotFoundError(str(meta_path))
        shard_meta = read_jsonl(meta_path)
        data = np.load(npz_path)
        for name in data.files:
            arrays_by_name[name].append(data[name])
        meta_rows.extend(shard_meta)
    order = np.argsort([int(row["row_index"]) for row in meta_rows])
    meta_sorted = [meta_rows[int(idx)] for idx in order]
    arrays = {}
    for name, chunks in arrays_by_name.items():
        joined = np.concatenate(chunks, axis=0)
        arrays[name] = joined[order]
    return meta_sorted, arrays


def feature_matrix_from_surface(surface_rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    matrix = []
    for row in surface_rows:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        matrix.append([float(features.get(name, 0.0) or 0.0) for name in names])
    return np.asarray(matrix, dtype=np.float32)


def surface_rows_for_eval(args: argparse.Namespace) -> List[Dict[str, Any]]:
    candidates_path = args.candidates
    if args.limit_steps > 0:
        limited_rows = read_jsonl(Path(args.candidates))[: args.limit_steps]
        tmp_path = Path(args.output_dir) / "surface_candidates.limit.jsonl"
        write_jsonl(tmp_path, limited_rows)
        candidates_path = str(tmp_path)
    ns = argparse.Namespace(
        candidates=candidates_path,
        test_data=args.test_data,
        crit_tasks=args.crit_tasks,
    )
    rows, _ = build_surface_rows(ns)
    rows.sort(key=lambda row: int(row.get("target_id", "0:0:0").split(":")[-1]) if False else (int(row.get("episode_order") or 0), int(row.get("step_idx") or 0)))
    by_key = {(str(row["episode_id"]), int(row["step_idx"])): row for row in rows}
    ordered = []
    candidate_rows = read_jsonl(Path(args.candidates))
    if args.limit_steps > 0:
        candidate_rows = candidate_rows[: args.limit_steps]
    for row in candidate_rows:
        key = (str(row.get("episode_id")), int(row.get("step_idx") or 0))
        if key in by_key:
            ordered.append(by_key[key])
    return ordered


def best_auc_value(item: Mapping[str, Any]) -> float:
    value = item.get("auc")
    return float(value) if value is not None else 0.0


def topk_table_from_scores(rows: Sequence[Mapping[str, Any]], scores: Sequence[float], budgets: Sequence[float], label_field: str) -> List[Dict[str, Any]]:
    y = np.asarray([int(row[label_field]) for row in rows], dtype=int)
    order = np.argsort(-np.asarray(scores, dtype=float))
    total_pos = int(np.sum(y == 1))
    out = []
    for budget in budgets:
        k = max(1, int(round(len(rows) * budget)))
        selected = order[:k]
        hit = int(np.sum(y[selected] == 1))
        out.append({
            "budget_fraction": budget,
            "selected_steps": k,
            "recall": hit / total_pos if total_pos else 0.0,
            "precision": hit / k if k else 0.0,
            "random_recall": budget,
            "random_precision": total_pos / len(rows) if rows else 0.0,
        })
    return out


def orient_score_for_auc(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    auc = auc_score(y.tolist(), score.tolist())
    if auc is not None and auc < 0.5:
        return -score
    return score


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_rows, arrays = load_activation_shards(output_dir, args.num_shards)
    surface_rows = surface_rows_for_eval(args)
    if len(surface_rows) != len(meta_rows):
        raise ValueError(f"surface/meta row mismatch: {len(surface_rows)} vs {len(meta_rows)}")
    labels = {
        "bottom2": np.asarray([int(row["bottom2"]) for row in meta_rows], dtype=int),
        "bottom1": np.asarray([int(row["bottom1"]) for row in meta_rows], dtype=int),
    }
    feature_names = sorted(name for name in surface_rows[0]["features"].keys() if feature_name_is_allowed(name))
    confidence_features = [
        name for name in feature_names
        if name in {
            "action_entropy", "type_entropy", "control_entropy", "one_minus_modal_action_frac", "one_minus_modal_type_frac", "one_minus_modal_control_frac",
            "sample_logprob_avg_mean", "sample_logprob_avg_std", "sample_logprob_avg_max", "sample_logprob_avg_min", "sample_logprob_avg_gap_top2",
            "sample_logprob_sum_mean", "sample_logprob_token_mean", "logprob_available_frac", "pred_text_len_mean", "pred_text_len_std",
        }
    ]
    surface_results = {}
    confidence_results = {}
    per_signal = {}
    for label_name, y in labels.items():
        rows_for_label = []
        for surf, meta in zip(surface_rows, meta_rows, strict=True):
            item = dict(surf)
            item["critical"] = bool(meta[label_name])
            rows_for_label.append(item)
        surface_results[label_name] = logistic_cv(rows_for_label, feature_names, args.folds, args.seed)
        confidence_results[label_name] = logistic_cv(rows_for_label, confidence_features, args.folds, args.seed)
        per_signal[label_name] = per_signal_metrics(rows_for_label, confidence_features)

    probe_results: Dict[str, List[Dict[str, Any]]] = {"bottom2": [], "bottom1": []}
    score_store: Dict[Tuple[str, str], np.ndarray] = {}
    confidence_matrix = feature_matrix_from_surface(surface_rows, confidence_features)
    for rep_name, raw in sorted(arrays.items()):
        seed = args.seed + (abs(hash(rep_name)) % 100000)
        x_proj = random_project(raw, args.probe_dim, seed)
        for label_name, y in labels.items():
            lin = logistic_cv_array(x_proj, y, args.folds, args.seed, l2=args.l2)
            combo = logistic_cv_array(np.concatenate([x_proj, confidence_matrix], axis=1), y, args.folds, args.seed, l2=args.l2)
            item = {
                "label": label_name,
                "representation": rep_name,
                "n": int(len(y)),
                "positives": int(np.sum(y == 1)),
                "linear_auc": lin["auc"],
                "linear_balanced_accuracy": lin["balanced_accuracy"],
                "confidence_plus_rep_auc": combo["auc"],
                "increment_over_confidence_auc": (combo["auc"] - confidence_results[label_name]["auc"]) if combo["auc"] is not None and confidence_results[label_name]["auc"] is not None else None,
            }
            probe_results[label_name].append(item)
            score_store[(label_name, rep_name)] = np.asarray(lin["scores"], dtype=float)
    for label_name in probe_results:
        probe_results[label_name].sort(key=lambda item: item["linear_auc"] if item["linear_auc"] is not None else -1.0, reverse=True)

    mlp_results: Dict[str, List[Dict[str, Any]]] = {"bottom2": [], "bottom1": []}
    for label_name in ("bottom2", "bottom1"):
        for item in probe_results[label_name][: args.mlp_top_k]:
            rep_name = item["representation"]
            seed = args.seed + 17 + (abs(hash(rep_name)) % 100000)
            x_proj = random_project(arrays[rep_name], args.probe_dim, seed)
            mlp = mlp_cv_array(x_proj, labels[label_name], args.folds, args.seed, args.mlp_epochs, args.mlp_hidden_dim)
            mlp_results[label_name].append({
                "label": label_name,
                "representation": rep_name,
                "mlp_auc": mlp["auc"],
                "mlp_balanced_accuracy": mlp["balanced_accuracy"],
            })

    primary = "bottom2"
    best = probe_results[primary][0]
    best_scores = orient_score_for_auc(labels[primary], score_store[(primary, best["representation"])]).astype(float)
    surface_scores = np.asarray(surface_results[primary]["scores"], dtype=float)
    confidence_scores = np.asarray(confidence_results[primary]["scores"], dtype=float)
    triage = {
        "probe": topk_table_from_scores(meta_rows, best_scores, BUDGETS, primary),
        "surface_logistic": topk_table_from_scores(meta_rows, orient_score_for_auc(labels[primary], surface_scores), BUDGETS, primary),
        "confidence_only": topk_table_from_scores(meta_rows, orient_score_for_auc(labels[primary], confidence_scores), BUDGETS, primary),
    }

    interpretability = []
    if best["linear_auc"] is not None and best["linear_auc"] >= 0.65:
        for signal in per_signal_metrics([{**row, "critical": bool(labels[primary][idx])} for idx, row in enumerate(surface_rows)], feature_names)[:30]:
            values = np.asarray([float(row["features"].get(signal["feature"], 0.0) or 0.0) for row in surface_rows], dtype=float)
            if np.std(values) > 1e-8:
                corr = float(np.corrcoef(best_scores, values)[0, 1])
            else:
                corr = 0.0
            interpretability.append({"feature": signal["feature"], "corr_with_probe_score": corr, "signal_oriented_auc": signal.get("best_oriented_auc")})
        interpretability.sort(key=lambda item: abs(float(item["corr_with_probe_score"])), reverse=True)

    best_auc = float(best.get("linear_auc") or 0.0)
    conf_auc = float(confidence_results[primary].get("auc") or 0.0)
    increment = float(best.get("increment_over_confidence_auc") or 0.0)
    probe_top20 = next(row for row in triage["probe"] if abs(row["budget_fraction"] - 0.20) < 1e-9)["recall"]
    surface_top20 = next(row for row in triage["surface_logistic"] if abs(row["budget_fraction"] - 0.20) < 1e-9)["recall"]
    if best_auc >= 0.75 and increment >= 0.03 and probe_top20 > max(surface_top20, BASELINE_SURFACE_TOP20_RECALL):
        gate = {
            "verdict": "CRITICAL STEPS ARE A REPRESENTATIONAL CLASS",
            "reason": "The best pre-decision activation probe is clearly above surface/confidence baselines and improves operational triage.",
        }
    elif best_auc >= 0.65 or (best_auc > BASELINE_SURFACE_AUC and increment <= 0.01):
        gate = {
            "verdict": "CONFIDENCE-ONLY / WEAK",
            "reason": "Representation separability is modest or adds little beyond confidence; it is not a clean operational class.",
        }
    else:
        gate = {
            "verdict": "INTERNALLY INSEPARABLE",
            "reason": "Pre-decision activation probes remain around weak surface/uncertainty baselines, so bottom-k critical steps do not form a robust internal class.",
        }

    per_step_rows = []
    ranks = np.argsort(-best_scores)
    rank_by_idx = {int(idx): rank + 1 for rank, idx in enumerate(ranks)}
    for idx, (meta, surf) in enumerate(zip(meta_rows, surface_rows, strict=True)):
        per_step_rows.append({
            "row_index": meta["row_index"],
            "target_id": meta["target_id"],
            "episode_id": meta["episode_id"],
            "step_idx": meta["step_idx"],
            "bottom2_critical": bool(meta["bottom2"]),
            "bottom1_critical": bool(meta["bottom1"]),
            "p_i_heldout_label_only": meta.get("p_i_heldout_label_only"),
            "best_probe_representation": best["representation"],
            "best_probe_score": float(best_scores[idx]),
            "best_probe_rank": rank_by_idx[idx],
            "best_probe_percentile": rank_by_idx[idx] / len(meta_rows),
            "surface_score": float(surface_scores[idx]),
            "confidence_score": float(confidence_scores[idx]),
            "prompt_token_count": meta.get("prompt_token_count"),
            "image_token_count": meta.get("image_token_count"),
            "text_token_count": meta.get("text_token_count"),
            "activation_artifact": "outputs/critstep_representation/activations/*.npz",
        })
    write_jsonl(output_dir / "per_step.jsonl", per_step_rows)

    summary = {
        "inputs": {"model": args.model, "candidates": args.candidates, "test_data": args.test_data, "crit_tasks": args.crit_tasks},
        "dataset": {"rows": len(meta_rows), "bottom2_critical": int(np.sum(labels["bottom2"])), "bottom1_critical": int(np.sum(labels["bottom1"]))},
        "guardrails": {
            "pre_decision_only": True,
            "activation_source": "Prompt forward pass with add_generation_prompt=True; no generated action/outcome/reward is included in activation input.",
            "layers_positions_swept": sorted(arrays.keys()),
            "random_projection_dim": args.probe_dim,
        },
        "baselines_reference": {"entropy_auc": BASELINE_ENTROPY_AUC, "surface_auc": BASELINE_SURFACE_AUC, "internal_uncertainty_auc": BASELINE_INTERNAL_UNCERTAINTY_AUC, "surface_top20_recall": BASELINE_SURFACE_TOP20_RECALL},
        "surface_results_full": {label: {"auc": surface_results[label]["auc"], "balanced_accuracy": surface_results[label]["balanced_accuracy"]} for label in surface_results},
        "confidence_results_full": {label: {"auc": confidence_results[label]["auc"], "balanced_accuracy": confidence_results[label]["balanced_accuracy"], "features": confidence_features} for label in confidence_results},
        "probe_results": probe_results,
        "mlp_results_top_configs": mlp_results,
        "triage": triage,
        "interpretability": interpretability[:20],
        "gate": gate,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "probe.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "best": best, "gate": gate}, indent=2, ensure_ascii=False), flush=True)


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def render_probe_table(rows: Sequence[Mapping[str, Any]], limit: int = 20) -> List[str]:
    lines = ["| rank | representation | linear AUC | balanced acc | rep+confidence AUC | increment over confidence |", "|---:|---|---:|---:|---:|---:|"]
    for rank, item in enumerate(rows[:limit], 1):
        inc = item.get("increment_over_confidence_auc")
        lines.append(f"| {rank} | `{item['representation']}` | {fmt_pct(item.get('linear_auc'))} | {fmt_pct(item.get('linear_balanced_accuracy'))} | {fmt_pct(item.get('confidence_plus_rep_auc'))} | {fmt_pct(inc)} |")
    return lines


def render_triage_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = ["| budget | selected | recall | precision | random recall |", "|---:|---:|---:|---:|---:|"]
    for item in rows:
        lines.append(f"| {fmt_pct(item['budget_fraction'])} | {item['selected_steps']} | {fmt_pct(item['recall'])} | {fmt_pct(item['precision'])} | {fmt_pct(item['random_recall'])} |")
    return lines


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines: List[str] = ["# Critical-Step Representation Probe", ""]
    lines.append("Probe-only diagnostic. Activations are pre-decision prompt forward states from the frozen GUI-360 SFT model; labels are bottom-k held-out p_i only.")
    lines.append("")
    ds = summary["dataset"]
    lines.append(f"Rows: `{ds['rows']}`. Bottom-2 critical: `{ds['bottom2_critical']}`. Bottom-1 critical: `{ds['bottom1_critical']}`.")
    lines.append("")
    lines.append("## Metric 1: Probe Separability")
    lines.append("")
    lines.append("Primary label: bottom-2 critical vs non-critical.")
    lines.extend(render_probe_table(summary["probe_results"]["bottom2"], limit=16))
    lines.append("")
    lines.append("Bottom-1 label check.")
    lines.extend(render_probe_table(summary["probe_results"]["bottom1"], limit=8))
    lines.append("")
    lines.append("## Baselines And Confidence Increment")
    lines.append("")
    lines.append("| predictor | bottom-2 AUC | bottom-1 AUC | note |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| surface logistic full-test | {fmt_pct(summary['surface_results_full']['bottom2']['auc'])} | {fmt_pct(summary['surface_results_full']['bottom1']['auc'])} | recomputed on same rows |")
    lines.append(f"| confidence-only full-test | {fmt_pct(summary['confidence_results_full']['bottom2']['auc'])} | {fmt_pct(summary['confidence_results_full']['bottom1']['auc'])} | entropy/logprob/disagreement subset |")
    best = summary["probe_results"]["bottom2"][0]
    lines.append(f"| best activation probe | {fmt_pct(best.get('linear_auc'))} | {fmt_pct(summary['probe_results']['bottom1'][0].get('linear_auc'))} | `{best['representation']}` |")
    lines.append("")
    lines.append("Reference baselines from prior work/spec: entropy AUC `58%`, surface/internal-uncertainty AUC `63-63.4%`, surface top-20 recall `33.5%`.")
    lines.append("")
    lines.append("## Nonlinear Probe")
    lines.append("")
    lines.append("| label | representation | MLP AUC | balanced acc |")
    lines.append("|---|---|---:|---:|")
    for label, rows in summary["mlp_results_top_configs"].items():
        for item in rows:
            lines.append(f"| {label} | `{item['representation']}` | {fmt_pct(item.get('mlp_auc'))} | {fmt_pct(item.get('mlp_balanced_accuracy'))} |")
    lines.append("")
    lines.append("## Metric 3: Operational Triage")
    lines.append("")
    lines.append("Probe triage:")
    lines.extend(render_triage_table(summary["triage"]["probe"]))
    lines.append("")
    lines.append("Surface logistic triage on same rows:")
    lines.extend(render_triage_table(summary["triage"]["surface_logistic"]))
    lines.append("")
    lines.append("Confidence-only triage on same rows:")
    lines.extend(render_triage_table(summary["triage"]["confidence_only"]))
    lines.append("")
    lines.append("## Metric 4: Probe Direction")
    lines.append("")
    if summary.get("interpretability"):
        lines.append("Top correlations between best probe score and GT-free surface features:")
        lines.append("")
        lines.append("| feature | corr with probe score | feature oriented AUC |")
        lines.append("|---|---:|---:|")
        for item in summary["interpretability"][:12]:
            lines.append(f"| `{item['feature']}` | {item['corr_with_probe_score']:.3f} | {fmt_pct(item.get('signal_oriented_auc'))} |")
    else:
        lines.append("Skipped: probe did not reach a separable regime where direction interpretation would be meaningful.")
    lines.append("")
    lines.append("## Leakage Audit")
    lines.append("")
    guard = summary["guardrails"]
    lines.append(f"- pre-decision only: `{guard['pre_decision_only']}`")
    lines.append(f"- activation source: {guard['activation_source']}")
    lines.append(f"- swept representations: `{len(guard['layers_positions_swept'])}` layer/position combinations")
    lines.append(f"- random projection dimension: `{guard['random_projection_dim']}`")
    lines.append("- labels p_i/bottom-k are used only as targets, not as probe inputs.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    gate = summary["gate"]
    lines.append(f"**{gate['verdict']}**")
    lines.append("")
    lines.append(gate["reason"])
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'probe.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append(f"- `{output_dir / 'activations'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["extract", "evaluate", "all"], default="evaluate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--crit-tasks", default=DEFAULT_CRIT_TASKS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--probe-dim", type=int, default=256)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--mlp-top-k", type=int, default=4)
    parser.add_argument("--mlp-epochs", type=int, default=40)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    args = parser.parse_args()
    if args.phase in {"extract", "all"}:
        extract_shard(args)
    if args.phase in {"evaluate", "all"}:
        evaluate(args)


if __name__ == "__main__":
    main()
