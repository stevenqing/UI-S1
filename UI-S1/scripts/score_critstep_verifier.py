#!/usr/bin/env python3
"""Score the critical-step verifier eval slice with a trained LoRA adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_verifier_full_action import ORACLE_CRITICAL_TSR_CEILING_PP, pct, projected_tsr_lift_pp  # noqa: E402


DEFAULT_BASE_MODEL = "outputs/critstep_verifier/gui360_fullparam_sft_step250_trainview"
DEFAULT_ADAPTER = "outputs/critstep_verifier/verifier_lora_qwen25vl_overnight"
DEFAULT_PER_STEP = "outputs/critstep_verifier/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_verifier/eval_overnight"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_image(path: str) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def prompt_without_image_marker(prompt: str) -> str:
    if prompt.startswith("<image>\n"):
        return prompt[len("<image>\n") :]
    return prompt.replace("<image>", "", 1).lstrip()


def candidate_prompt_from_step(step: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    from scripts.critstep_verifier_full_action import verifier_prompt

    return verifier_prompt(
        str(step.get("instruction") or ""),
        step.get("history") if isinstance(step.get("history"), list) else [],
        candidate.get("action") if isinstance(candidate.get("action"), dict) else {},
        candidate.get("control") if isinstance(candidate.get("control"), dict) else {},
    )


def chat_text(processor: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_without_image_marker(prompt)}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prefix_text(processor: Any, prompt: str, prefix: str = "VERDICT:") -> str:
    return chat_text(processor, prompt) + prefix


def token_ids_for(processor: Any, text: str) -> List[int]:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"empty tokenization for {text!r}")
    return list(ids)


def logsumexp(values: Sequence[float]) -> float:
    high = max(values)
    return high + math.log(sum(math.exp(v - high) for v in values))


def score_batch(
    *,
    model: Any,
    processor: Any,
    prompts: Sequence[str],
    image_paths: Sequence[str],
    correct_token_ids: Sequence[int],
    incorrect_token_ids: Sequence[int],
    device: torch.device,
) -> List[Dict[str, float]]:
    texts = [prefix_text(processor, prompt) for prompt in prompts]
    images = [load_image(path) for path in image_paths]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1).detach().cpu()
    out = []
    for row in log_probs:
        correct_lp = logsumexp([float(row[idx]) for idx in correct_token_ids])
        incorrect_lp = logsumexp([float(row[idx]) for idx in incorrect_token_ids])
        denom = logsumexp([correct_lp, incorrect_lp])
        out.append({
            "verifier_logprob_correct": correct_lp,
            "verifier_logprob_incorrect": incorrect_lp,
            "verifier_score": math.exp(correct_lp - denom),
            "verifier_margin": correct_lp - incorrect_lp,
        })
    return out


def assign_scores_to_steps(steps: List[Dict[str, Any]], scored: Dict[Tuple[str, str], Dict[str, float]]) -> List[Dict[str, Any]]:
    output_steps = []
    for step in steps:
        step = dict(step)
        candidates = []
        best_candidate = None
        first_yes_candidate = None
        greedy_score = None
        best_correct_score = None
        for candidate in step.get("candidates", []):
            candidate = dict(candidate)
            key = (str(step["target_id"]), str(candidate["candidate_id"]))
            score = scored[key]
            candidate.update(score)
            candidates.append(candidate)
            if candidate.get("source") == "greedy":
                greedy_score = score["verifier_score"]
            if candidate.get("is_correct"):
                best_correct_score = max(best_correct_score or float("-inf"), score["verifier_score"])
            if best_candidate is None or score["verifier_score"] > best_candidate["verifier_score"]:
                best_candidate = candidate
            if first_yes_candidate is None and score["verifier_score"] >= 0.5:
                first_yes_candidate = candidate
        step["candidates"] = candidates
        step["verifier_candidate_id"] = best_candidate["candidate_id"] if best_candidate else None
        step["verifier_correct"] = bool(best_candidate and best_candidate.get("is_correct"))
        step["verifier_score"] = best_candidate["verifier_score"] if best_candidate else None
        step["first_yes_candidate_id"] = first_yes_candidate["candidate_id"] if first_yes_candidate else None
        step["first_yes_correct"] = bool(first_yes_candidate and first_yes_candidate.get("is_correct"))
        step["first_yes_score"] = first_yes_candidate["verifier_score"] if first_yes_candidate else None
        step["best_correct_score"] = best_correct_score
        step["greedy_score"] = greedy_score
        step["greedy_rejected_by_verifier"] = bool(best_correct_score is not None and greedy_score is not None and best_correct_score > greedy_score)
        output_steps.append(step)
    return output_steps


def fraction(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    if not rows:
        return None
    values = [row.get(field) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(1 for value in values if value) / len(rows)


def summarize_by(rows: Sequence[Mapping[str, Any]], group_field: str, metric_fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field))].append(row)
    out = {}
    for group, group_rows in sorted(grouped.items()):
        out[group] = {"n": len(group_rows)}
        for metric in metric_fields:
            out[group][metric] = fraction(group_rows, metric)
    return out


def gate(summary: Mapping[str, Any]) -> str:
    acc = summary["selection_accuracy"].get("verifier_argmax") or 0.0
    first = summary["selection_accuracy"].get("sample_order_first") or 0.0
    deep = summary["depth_stratified"].get("deep_21_50", {}).get("verifier_correct") or 0.0
    reject = summary.get("reject_greedy_rate") or 0.0
    if acc > first + 0.10 and deep >= 0.30 and reject >= 0.50:
        return "VERIFIER EFFECTIVE"
    if acc > first + 0.05 and deep < 0.20:
        return "VERIFIER SHALLOW-ONLY"
    return "VERIFIER INEFFECTIVE"


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Critical-Step Full-Action Verifier Evaluation", ""]
    lines.append("## 3.1 Selection Accuracy")
    lines.append("")
    lines.append("| selector | accuracy | projected TSR lift proxy | fraction of +23.77pp ceiling |")
    lines.append("|---|---:|---:|---:|")
    for selector, accuracy in summary["selection_accuracy"].items():
        lift = projected_tsr_lift_pp(accuracy, summary["recoverable_fraction_primary"]) if accuracy is not None else None
        ceiling_frac = lift / ORACLE_CRITICAL_TSR_CEILING_PP if lift is not None else None
        acc_text = f"{accuracy*100:.2f}%" if accuracy is not None else "NA"
        lift_text = f"{lift:.2f}pp" if lift is not None else "NA"
        frac_text = f"{ceiling_frac*100:.2f}%" if ceiling_frac is not None else "NA"
        lines.append(f"| {selector} | {acc_text} | {lift_text} | {frac_text} |")
    lines.append("")
    lines.append("## 3.1 Per-Subset Accuracy")
    lines.append("")
    lines.append("| subset | n | verifier argmax | first-yes | first sample |")
    lines.append("|---|---:|---:|---:|---:|")
    for subset, item in summary["subset_stratified"].items():
        lines.append(f"| {subset} | {item['n']} | {item['verifier_correct']*100:.2f}% | {item['first_yes_correct']*100:.2f}% | {item['first_sample_correct']*100:.2f}% |")
    lines.append("")
    lines.append("## 3.2 Depth-Stratified Recovery")
    lines.append("")
    lines.append("| depth bin | n | verifier argmax | first-yes | first sample |")
    lines.append("|---|---:|---:|---:|---:|")
    for depth, item in summary["depth_stratified"].items():
        lines.append(f"| {depth} | {item['n']} | {item['verifier_correct']*100:.2f}% | {item['first_yes_correct']*100:.2f}% | {item['first_sample_correct']*100:.2f}% |")
    lines.append("")
    lines.append("## 3.3 Reject-The-Distractor")
    lines.append("")
    lines.append(f"Reject-greedy rate: `{summary['reject_greedy_rate']*100:.2f}%`")
    lines.append("")
    lines.append("## 3.4 Compound Projection")
    lines.append("")
    lines.append(f"Recoverable@50 primary pool: `{summary['n_recoverable_primary']} / {summary['n_primary_failures']}` ({summary['recoverable_fraction_primary']*100:.2f}%).")
    lines.append(f"Verifier projected TSR lift proxy: `{summary['projected_tsr_lift_pp']:.2f}pp`, `{summary['projected_ceiling_fraction']*100:.2f}%` of the +23.77pp oracle-critical ceiling.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']}**")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'verifier_eval.md'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append(f"- `{output_dir / 'candidate_scores.jsonl'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def build_jobs(steps: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    jobs = []
    for step in steps:
        for candidate in step.get("candidates", []):
            jobs.append({
                "target_id": step["target_id"],
                "candidate_id": candidate["candidate_id"],
                "prompt": candidate_prompt_from_step(step, candidate),
                "screenshot": step["screenshot"],
                "is_correct": candidate.get("is_correct"),
                "source": candidate.get("source"),
            })
    return jobs


def summarize_scores(
    *,
    steps: List[Dict[str, Any]],
    score_rows: List[Dict[str, Any]],
    output_dir: Path,
    base_model: str,
    adapter: str,
    per_step: str,
) -> Dict[str, Any]:
    scored = {
        (str(row["target_id"]), str(row["candidate_id"])): {
            "verifier_logprob_correct": row["verifier_logprob_correct"],
            "verifier_logprob_incorrect": row["verifier_logprob_incorrect"],
            "verifier_score": row["verifier_score"],
            "verifier_margin": row["verifier_margin"],
        }
        for row in score_rows
    }
    scored_steps = assign_scores_to_steps(steps, scored)
    write_jsonl(output_dir / "candidate_scores.jsonl", score_rows)
    write_jsonl(output_dir / "per_step.jsonl", scored_steps)

    selection_accuracy = {
        "oracle_in_pool": 1.0,
        "greedy": fraction(scored_steps, "greedy_correct"),
        "sample_order_first": fraction(scored_steps, "first_sample_correct"),
        "verifier_argmax": fraction(scored_steps, "verifier_correct"),
        "verifier_first_yes": fraction(scored_steps, "first_yes_correct"),
    }
    verifier_acc = selection_accuracy["verifier_argmax"] or 0.0
    recoverable_fraction = 488 / 862
    lift = projected_tsr_lift_pp(verifier_acc, recoverable_fraction) or 0.0
    summary = {
        "base_model": base_model,
        "adapter": adapter,
        "per_step": per_step,
        "n_steps": len(scored_steps),
        "n_candidates": len(score_rows),
        "n_primary_failures": 862,
        "n_recoverable_primary": 488,
        "recoverable_fraction_primary": recoverable_fraction,
        "selection_accuracy": selection_accuracy,
        "subset_stratified": summarize_by(scored_steps, "subset", ["verifier_correct", "first_yes_correct", "first_sample_correct"]),
        "depth_stratified": summarize_by(scored_steps, "depth_bin", ["verifier_correct", "first_yes_correct", "first_sample_correct"]),
        "reject_greedy_rate": fraction(scored_steps, "greedy_rejected_by_verifier"),
        "projected_tsr_lift_pp": lift,
        "projected_ceiling_fraction": lift / ORACLE_CRITICAL_TSR_CEILING_PP,
    }
    summary["gate"] = gate(summary)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "verifier_eval.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    return summary


def merge_shards(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    steps = read_jsonl(Path(args.per_step))
    if args.limit_steps > 0:
        steps = steps[: args.limit_steps]
    jobs = build_jobs(steps)
    shard_paths = sorted(output_dir.glob(f"candidate_scores.shard_*_of_{args.num_shards:02d}.jsonl"))
    if len(shard_paths) != args.num_shards:
        raise FileNotFoundError(f"expected {args.num_shards} shard files, found {len(shard_paths)} in {output_dir}")
    score_rows: List[Dict[str, Any]] = []
    seen = set()
    for path in shard_paths:
        for row in read_jsonl(path):
            key = (str(row["target_id"]), str(row["candidate_id"]))
            if key in seen:
                raise ValueError(f"duplicate score row for {key}")
            seen.add(key)
            score_rows.append(row)
    expected = {(str(job["target_id"]), str(job["candidate_id"])) for job in jobs}
    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        raise ValueError(f"score row mismatch: missing={len(missing)} extra={len(extra)}")
    score_rows.sort(key=lambda row: int(row.get("job_index", 10**12)))
    summary = summarize_scores(
        steps=steps,
        score_rows=score_rows,
        output_dir=output_dir,
        base_model=args.base_model,
        adapter=args.adapter,
        per_step=args.per_step,
    )
    print(json.dumps({"output_dir": str(output_dir), "gate": summary["gate"], "verifier_acc": summary["selection_accuracy"]["verifier_argmax"], "reject_greedy": summary["reject_greedy_rate"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--per-step", default=DEFAULT_PER_STEP)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        merge_shards(args)
        return
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"loading processor/model base={args.base_model} adapter={args.adapter} device={device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval().to(device)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    correct_ids = token_ids_for(processor, " correct") + token_ids_for(processor, "correct")
    incorrect_ids = token_ids_for(processor, " incorrect") + token_ids_for(processor, "incorrect")
    correct_ids = sorted(set(correct_ids))
    incorrect_ids = sorted(set(incorrect_ids))
    print(f"correct_ids={correct_ids} incorrect_ids={incorrect_ids}", flush=True)

    steps = read_jsonl(Path(args.per_step))
    if args.limit_steps > 0:
        steps = steps[: args.limit_steps]
    all_jobs = build_jobs(steps)
    indexed_jobs = [(idx, job) for idx, job in enumerate(all_jobs) if idx % args.num_shards == args.shard_index]
    score_path = output_dir / "candidate_scores.jsonl"
    if args.num_shards > 1:
        score_path = output_dir / f"candidate_scores.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.jsonl"
    scored: Dict[Tuple[str, str], Dict[str, float]] = {}
    score_rows = []
    seen_keys = set()
    if args.resume and score_path.exists():
        score_rows = read_jsonl(score_path)
        for row in score_rows:
            key = (str(row["target_id"]), str(row["candidate_id"]))
            seen_keys.add(key)
            scored[key] = {
                "verifier_logprob_correct": row["verifier_logprob_correct"],
                "verifier_logprob_incorrect": row["verifier_logprob_incorrect"],
                "verifier_score": row["verifier_score"],
                "verifier_margin": row["verifier_margin"],
            }
        indexed_jobs = [(idx, job) for idx, job in indexed_jobs if (str(job["target_id"]), str(job["candidate_id"])) not in seen_keys]
        print(f"resumed {len(score_rows)} existing scores from {score_path}", flush=True)
    jobs = [job for _, job in indexed_jobs]
    job_indices = [idx for idx, _ in indexed_jobs]
    if args.num_shards > 1:
        print(f"shard {args.shard_index}/{args.num_shards}: scoring {len(jobs)} remaining candidates from {len(all_jobs)} total", flush=True)
    batch_size = max(1, args.batch_size)
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        batch_indices = job_indices[start : start + batch_size]
        scores = score_batch(
            model=model,
            processor=processor,
            prompts=[job["prompt"] for job in batch],
            image_paths=[job["screenshot"] for job in batch],
            correct_token_ids=correct_ids,
            incorrect_token_ids=incorrect_ids,
            device=device,
        )
        for job_index, job, score in zip(batch_indices, batch, scores, strict=True):
            key = (str(job["target_id"]), str(job["candidate_id"]))
            scored[key] = score
            score_rows.append({"job_index": job_index} | {key_: job[key_] for key_ in ["target_id", "candidate_id", "is_correct", "source"]} | score)
        done = min(start + batch_size, len(jobs))
        if done % 100 == 0 or done == len(jobs):
            print(f"scored {done}/{len(jobs)} candidates", flush=True)
            write_jsonl(score_path, score_rows)
    write_jsonl(score_path, score_rows)
    if args.num_shards > 1:
        print(json.dumps({"score_path": str(score_path), "n_candidates": len(score_rows)}, indent=2), flush=True)
        return
    summary = summarize_scores(
        steps=steps,
        score_rows=score_rows,
        output_dir=output_dir,
        base_model=args.base_model,
        adapter=args.adapter,
        per_step=args.per_step,
    )
    print(json.dumps({"output_dir": str(output_dir), "gate": summary["gate"], "verifier_acc": summary["selection_accuracy"]["verifier_argmax"], "reject_greedy": summary["reject_greedy_rate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()