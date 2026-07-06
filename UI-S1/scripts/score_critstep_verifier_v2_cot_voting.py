#!/usr/bin/env python3
"""Score Stage-1 GenRM-CoT verifier with inference-time voting."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_verifier_full_action import (  # noqa: E402
    ORACLE_CRITICAL_TSR_CEILING_PP,
    projected_tsr_lift_pp,
    render_candidate_control,
    render_history,
)


DEFAULT_BASE_MODEL = "outputs/critstep_verifier_v2/gui360_fullparam_sft_step250_trainview"
DEFAULT_ADAPTER = "outputs/critstep_verifier_v2/stage1_genrm_cot_lora"
DEFAULT_PER_STEP = "outputs/critstep_verifier/per_step.jsonl"
DEFAULT_POINTWISE_SUMMARY = "outputs/critstep_verifier/eval_overnight/summary.json"
DEFAULT_OUTPUT_DIR = "outputs/critstep_verifier_v2/stage1_eval_200"


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


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def stage1_prompt(goal: str, history: Sequence[str], candidate_action: Mapping[str, Any], control_info: Mapping[str, Any]) -> str:
    action_text = json.dumps(candidate_action, ensure_ascii=False, sort_keys=True)
    return (
        "<image>\n"
        "You are a GenRM-style generative verifier for GUI actions. Given the current screenshot, user instruction, action history, "
        "and exactly one candidate action, decide whether the candidate is the correct next FULL ACTION.\n\n"
        "Judge the whole action: action type, target element/control if any, coordinates if relevant, and typed/key content if relevant.\n"
        "Use only the screenshot, instruction, history, candidate action, and candidate UIA metadata below. Do not assume candidate frequency or rank.\n\n"
        f"Instruction:\n{goal}\n\n"
        f"Action history:\n{render_history(history)}\n\n"
        f"Candidate action JSON:\n{action_text}\n\n"
        f"Candidate UIA control metadata:\n{render_candidate_control(control_info)}\n\n"
        "Return short verification reasoning under exactly these three lines:\n"
        "Type: does the candidate action type match the intended next step?\n"
        "Target: does the target UIA control text/type/geometry match the instruction referent?\n"
        "Content: if the action types text or presses a key, is the content right?\n"
        "Then finish with a final line exactly one of:\n"
        "VERDICT: Yes\n"
        "VERDICT: No"
    )


def prompt_without_image_marker(prompt: str) -> str:
    if prompt.startswith("<image>\n"):
        return prompt[len("<image>\n") :]
    return prompt.replace("<image>", "", 1).lstrip()


def chat_text(processor: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_without_image_marker(prompt)}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def token_ids_for(processor: Any, texts: Sequence[str]) -> List[int]:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ids = []
    for text in texts:
        tokenized = tokenizer(text, add_special_tokens=False).input_ids
        ids.extend(int(item) for item in tokenized)
    return sorted(set(ids))


def logsumexp_from_logits(logits: torch.Tensor, token_ids: Sequence[int]) -> float:
    values = logits[list(token_ids)].float()
    return float(torch.logsumexp(values, dim=0))


def yes_probability(logits: torch.Tensor, yes_ids: Sequence[int], no_ids: Sequence[int]) -> float:
    yes_lp = logsumexp_from_logits(logits, yes_ids)
    no_lp = logsumexp_from_logits(logits, no_ids)
    high = max(yes_lp, no_lp)
    denom = high + math.log(math.exp(yes_lp - high) + math.exp(no_lp - high))
    return math.exp(yes_lp - denom)


def parse_verdict(text: str) -> Optional[str]:
    match = re.search(r"VERDICT\s*:\s*(Yes|No)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None


def find_verdict_step(tokenizer: Any, generated_ids: Sequence[int], yes_ids: Sequence[int], no_ids: Sequence[int]) -> Optional[int]:
    text_so_far = ""
    verdict_seen = False
    yes_no = set(yes_ids) | set(no_ids)
    for idx, token_id in enumerate(generated_ids):
        text_so_far += tokenizer.decode([int(token_id)], skip_special_tokens=True)
        if "VERDICT" in text_so_far.upper():
            verdict_seen = True
        if verdict_seen and int(token_id) in yes_no:
            return idx
    for idx, token_id in enumerate(generated_ids):
        if int(token_id) in yes_no:
            return idx
    return None


def score_generation_batch(
    *,
    model: Any,
    processor: Any,
    prompts: Sequence[str],
    image_paths: Sequence[str],
    vote_chunk: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    yes_ids: Sequence[int],
    no_ids: Sequence[int],
    device: torch.device,
    score_mode: str,
) -> List[List[Dict[str, Any]]]:
    tokenizer = processor.tokenizer
    texts = [chat_text(processor, prompt) for prompt in prompts]
    images = [load_image(path) for path in image_paths]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])
    output_scores = score_mode == "token_prob"
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=vote_chunk,
            output_scores=output_scores,
            return_dict_in_generate=output_scores,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if output_scores:
        sequences = outputs.sequences.detach().cpu()
        score_tensors = [score.detach().cpu() for score in outputs.scores]
    else:
        sequences = outputs.detach().cpu()
        score_tensors = []
    grouped: List[List[Dict[str, Any]]] = [[] for _ in prompts]
    for seq_idx, sequence in enumerate(sequences):
        prompt_idx = seq_idx // vote_chunk
        generated_ids = [int(token_id) for token_id in sequence[input_len:].tolist()]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        verdict = parse_verdict(text)
        step_idx = find_verdict_step(tokenizer, generated_ids, yes_ids, no_ids)
        if output_scores and step_idx is not None and step_idx < len(score_tensors):
            prob_yes = yes_probability(score_tensors[step_idx][seq_idx], yes_ids, no_ids)
        elif verdict == "Yes":
            prob_yes = 1.0
        elif verdict == "No":
            prob_yes = 0.0
        else:
            prob_yes = 0.5
        grouped[prompt_idx].append({"yes_prob": prob_yes, "verdict": verdict, "text_preview": text[:400]})
    return grouped


def candidate_distinct_key(candidate: Mapping[str, Any]) -> str:
    control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
    payload = {
        "action_signature": candidate.get("action_signature"),
        "pred_type": candidate.get("pred_type"),
        "control_key": control.get("key"),
        "control_assignment": control.get("assignment"),
        "control_rect": control.get("rect"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build_jobs(steps: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], str]]:
    jobs: List[Dict[str, Any]] = []
    candidate_to_distinct: Dict[Tuple[str, str], str] = {}
    for step in steps:
        target_id = str(step["target_id"])
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for candidate in step.get("candidates", []):
            distinct_key = candidate_distinct_key(candidate)
            groups[distinct_key].append(candidate)
            candidate_to_distinct[(target_id, str(candidate["candidate_id"]))] = distinct_key
        for group_index, (distinct_key, candidates) in enumerate(sorted(groups.items())):
            representative = candidates[0]
            jobs.append({
                "target_id": target_id,
                "distinct_key": distinct_key,
                "job_id": f"{target_id}::distinct_{group_index:02d}",
                "candidate_ids": [str(candidate["candidate_id"]) for candidate in candidates],
                "representative_candidate_id": str(representative["candidate_id"]),
                "is_correct": any(bool(candidate.get("is_correct")) for candidate in candidates),
                "source": representative.get("source"),
                "prompt": stage1_prompt(
                    str(step.get("instruction") or ""),
                    step.get("history") if isinstance(step.get("history"), list) else [],
                    representative.get("action") if isinstance(representative.get("action"), dict) else {},
                    representative.get("control") if isinstance(representative.get("control"), dict) else {},
                ),
                "screenshot": step.get("screenshot"),
            })
    return jobs, candidate_to_distinct


def mean_prefix(values: Sequence[float], k: int) -> Optional[float]:
    if len(values) < k:
        return None
    return sum(values[:k]) / k


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
    out: Dict[str, Dict[str, Any]] = {}
    for group, group_rows in sorted(grouped.items()):
        out[group] = {"n": len(group_rows)}
        for metric in metric_fields:
            out[group][metric] = fraction(group_rows, metric)
    return out


def assign_scores(
    *,
    steps: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    vote_ks: Sequence[int],
) -> List[Dict[str, Any]]:
    by_target: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for job in jobs:
        by_target[str(job["target_id"])].append(job)
    scored_steps: List[Dict[str, Any]] = []
    for step in steps:
        target_id = str(step["target_id"])
        step_jobs = by_target[target_id]
        scored_step = dict(step)
        candidates = []
        for candidate in step.get("candidates", []):
            candidate = dict(candidate)
            distinct_key = candidate_distinct_key(candidate)
            match = next(job for job in step_jobs if job["distinct_key"] == distinct_key)
            candidate["stage1_distinct_key"] = distinct_key
            for k in vote_ks:
                candidate[f"stage1_score_k{k}"] = match.get(f"score_k{k}")
            candidates.append(candidate)
        scored_step["candidates"] = candidates
        for k in vote_ks:
            best = max(step_jobs, key=lambda job: (job.get(f"score_k{k}") if job.get(f"score_k{k}") is not None else -1.0))
            correct_scores = [job.get(f"score_k{k}") for job in step_jobs if job.get("is_correct") and job.get(f"score_k{k}") is not None]
            greedy_job = next((job for job in step_jobs if "greedy" in set(job.get("candidate_ids", []))), None)
            greedy_score = greedy_job.get(f"score_k{k}") if greedy_job else None
            best_correct_score = max(correct_scores) if correct_scores else None
            scored_step[f"stage1_k{k}_candidate_id"] = best.get("representative_candidate_id")
            scored_step[f"stage1_k{k}_distinct_key"] = best.get("distinct_key")
            scored_step[f"stage1_k{k}_score"] = best.get(f"score_k{k}")
            scored_step[f"stage1_k{k}_correct"] = bool(best.get("is_correct"))
            scored_step[f"stage1_k{k}_greedy_rejected"] = bool(best_correct_score is not None and greedy_score is not None and best_correct_score > greedy_score)
        scored_steps.append(scored_step)
    return scored_steps


def stage1_gate(summary: Mapping[str, Any]) -> str:
    best_acc = summary.get("best_accuracy") or 0.0
    best_deep = summary.get("best_depth_accuracy", {}).get("deep_21_50") or 0.0
    pointwise_deep = summary.get("pointwise_baseline", {}).get("deep_21_50") or 0.0
    if best_acc >= 0.40 or (best_deep >= 0.25 and best_deep >= pointwise_deep + 0.08):
        return "VERIFIER SAVABLE"
    return "TASK-INTRINSIC HARD"


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Stage 1 GenRM-CoT + Voting", ""]
    lines.append("## Selection Accuracy Vs K")
    lines.append("")
    lines.append("| selector | accuracy | projected TSR lift proxy | fraction of +23.77pp ceiling |")
    lines.append("|---|---:|---:|---:|")
    for name, accuracy in summary["selection_accuracy"].items():
        lift = projected_tsr_lift_pp(accuracy, summary["recoverable_fraction_primary"]) if accuracy is not None else None
        frac = lift / ORACLE_CRITICAL_TSR_CEILING_PP if lift is not None else None
        lines.append(f"| {name} | {accuracy*100:.2f}% | {lift:.2f}pp | {frac*100:.2f}% |" if accuracy is not None else f"| {name} | NA | NA | NA |")
    lines.append("")
    best_k = summary["best_k"]
    lines.append(f"Best K: `{best_k}`")
    lines.append("")
    lines.append("## Depth-Stratified At Best K")
    lines.append("")
    lines.append("| depth bin | n | CoT voting | previous pointwise |")
    lines.append("|---|---:|---:|---:|")
    pointwise_depth = summary.get("pointwise_baseline", {})
    for depth, item in summary["depth_stratified_best"].items():
        prev = pointwise_depth.get(depth)
        prev_text = f"{prev*100:.2f}%" if prev is not None else "NA"
        lines.append(f"| {depth} | {item['n']} | {item[f'stage1_k{best_k}_correct']*100:.2f}% | {prev_text} |")
    lines.append("")
    lines.append("## Per-Subset At Best K")
    lines.append("")
    lines.append("| subset | n | CoT voting |")
    lines.append("|---|---:|---:|")
    for subset, item in summary["subset_stratified_best"].items():
        lines.append(f"| {subset} | {item['n']} | {item[f'stage1_k{best_k}_correct']*100:.2f}% |")
    lines.append("")
    lines.append("## Reject-Greedy")
    lines.append("")
    lines.append("| K | reject-greedy rate |")
    lines.append("|---:|---:|")
    for k, value in summary["reject_greedy_by_k"].items():
        lines.append(f"| {k} | {value*100:.2f}% |")
    lines.append("")
    lines.append("## Stage-1 Gate")
    lines.append("")
    lines.append(f"**{summary['stage1_gate']}**")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'stage1_genrm_cot.md'}`")
    lines.append(f"- `{output_dir / 'stage1_summary.json'}`")
    lines.append(f"- `{output_dir / 'stage1_per_step.jsonl'}`")
    lines.append(f"- `{output_dir / 'stage1_candidate_scores.jsonl'}`")
    lines.append("")
    lines.append("STOP for review. Do not run Stage 2 unless Stage 1 is `VERIFIER SAVABLE` and the owner approves.")
    return "\n".join(lines) + "\n"


def summarize(
    *,
    steps: List[Dict[str, Any]],
    score_rows: List[Dict[str, Any]],
    output_dir: Path,
    vote_ks: Sequence[int],
    pointwise_summary_path: Path,
    base_model: str,
    adapter: str,
) -> Dict[str, Any]:
    scored_steps = assign_scores(steps=steps, jobs=score_rows, vote_ks=vote_ks)
    write_jsonl(output_dir / "stage1_candidate_scores.jsonl", score_rows)
    write_jsonl(output_dir / "stage1_per_step.jsonl", scored_steps)
    pointwise = load_json(pointwise_summary_path, {}) or {}
    pointwise_depth = {}
    for depth, item in (pointwise.get("depth_stratified") or {}).items():
        pointwise_depth[depth] = item.get("verifier_correct")
    selection_accuracy: Dict[str, Optional[float]] = {
        "oracle_in_pool": 1.0,
        "greedy": fraction(scored_steps, "greedy_correct"),
        "sample_order_first": fraction(scored_steps, "first_sample_correct"),
        "previous_pointwise_verifier": (pointwise.get("selection_accuracy") or {}).get("verifier_argmax"),
    }
    reject_greedy_by_k = {}
    for k in vote_ks:
        selection_accuracy[f"cot_vote_k{k}"] = fraction(scored_steps, f"stage1_k{k}_correct")
        reject_greedy_by_k[str(k)] = fraction(scored_steps, f"stage1_k{k}_greedy_rejected")
    best_k = max(vote_ks, key=lambda k: selection_accuracy[f"cot_vote_k{k}"] or 0.0)
    best_accuracy = selection_accuracy[f"cot_vote_k{best_k}"] or 0.0
    depth_best = summarize_by(scored_steps, "depth_bin", [f"stage1_k{best_k}_correct"])
    subset_best = summarize_by(scored_steps, "subset", [f"stage1_k{best_k}_correct"])
    best_depth_accuracy = {key: value[f"stage1_k{best_k}_correct"] for key, value in depth_best.items()}
    recoverable_fraction = 488 / 862
    lift = projected_tsr_lift_pp(best_accuracy, recoverable_fraction) or 0.0
    summary = {
        "base_model": base_model,
        "adapter": adapter,
        "n_steps": len(scored_steps),
        "n_distinct_candidates": len(score_rows),
        "vote_ks": list(vote_ks),
        "best_k": best_k,
        "best_accuracy": best_accuracy,
        "best_depth_accuracy": best_depth_accuracy,
        "pointwise_baseline": {
            "overall": (pointwise.get("selection_accuracy") or {}).get("verifier_argmax"),
            **pointwise_depth,
        },
        "n_primary_failures": 862,
        "n_recoverable_primary": 488,
        "recoverable_fraction_primary": recoverable_fraction,
        "selection_accuracy": selection_accuracy,
        "depth_stratified_best": depth_best,
        "subset_stratified_best": subset_best,
        "reject_greedy_by_k": reject_greedy_by_k,
        "projected_tsr_lift_pp_best": lift,
        "projected_ceiling_fraction_best": lift / ORACLE_CRITICAL_TSR_CEILING_PP,
    }
    summary["stage1_gate"] = stage1_gate(summary)
    write_json(output_dir / "stage1_summary.json", summary)
    (output_dir / "stage1_genrm_cot.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    return summary


def parse_vote_ks(text: str) -> List[int]:
    values = sorted({int(item) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("empty --vote-ks")
    return values


def score_rows_for_jobs(
    *,
    model: Any,
    processor: Any,
    jobs: Sequence[Mapping[str, Any]],
    job_indices: Sequence[int],
    score_rows: List[Dict[str, Any]],
    score_path: Path,
    vote_ks: Sequence[int],
    batch_size: int,
    vote_chunk: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    yes_ids: Sequence[int],
    no_ids: Sequence[int],
    device: torch.device,
    score_mode: str,
) -> List[Dict[str, Any]]:
    max_k = max(vote_ks)
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        batch_indices = job_indices[start : start + batch_size]
        votes_by_job: List[List[Dict[str, Any]]] = [[] for _ in batch]
        while min(len(votes) for votes in votes_by_job) < max_k:
            current_chunk = min(vote_chunk, max_k - min(len(votes) for votes in votes_by_job))
            chunk_votes = score_generation_batch(
                model=model,
                processor=processor,
                prompts=[str(job["prompt"]) for job in batch],
                image_paths=[str(job["screenshot"]) for job in batch],
                vote_chunk=current_chunk,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                yes_ids=yes_ids,
                no_ids=no_ids,
                device=device,
                score_mode=score_mode,
            )
            for idx, votes in enumerate(chunk_votes):
                votes_by_job[idx].extend(votes)
        for job_index, job, votes in zip(batch_indices, batch, votes_by_job, strict=True):
            yes_probs = [float(vote["yes_prob"]) for vote in votes[:max_k]]
            parsed = [vote.get("verdict") for vote in votes[:max_k]]
            row = {
                "job_index": int(job_index),
                "target_id": job["target_id"],
                "job_id": job["job_id"],
                "distinct_key": job["distinct_key"],
                "candidate_ids": job["candidate_ids"],
                "representative_candidate_id": job["representative_candidate_id"],
                "is_correct": job["is_correct"],
                "source": job.get("source"),
                "score_mode": score_mode,
                "yes_probs": yes_probs,
                "parsed_verdicts": parsed,
                "text_previews": [vote.get("text_preview") for vote in votes[: min(2, len(votes))]],
            }
            for k in vote_ks:
                row[f"score_k{k}"] = mean_prefix(yes_probs, k)
                row[f"yes_vote_fraction_k{k}"] = sum(1 for verdict in parsed[:k] if verdict == "Yes") / k
            score_rows.append(row)
        done = min(start + batch_size, len(jobs))
        if done % 20 == 0 or done == len(jobs):
            print(f"scored {done}/{len(jobs)} distinct candidates", flush=True)
            write_jsonl(score_path, score_rows)
    write_jsonl(score_path, score_rows)
    return score_rows


def merge_shards(args: argparse.Namespace, vote_ks: Sequence[int]) -> None:
    output_dir = Path(args.output_dir)
    steps = read_jsonl(Path(args.per_step))
    if args.limit_steps > 0:
        steps = steps[: args.limit_steps]
    jobs, _ = build_jobs(steps)
    shard_paths = sorted(output_dir.glob(f"stage1_candidate_scores.shard_*_of_{args.num_shards:02d}.jsonl"))
    if len(shard_paths) != args.num_shards:
        raise FileNotFoundError(f"expected {args.num_shards} shard files, found {len(shard_paths)}")
    rows: List[Dict[str, Any]] = []
    seen = set()
    for path in shard_paths:
        for row in read_jsonl(path):
            key = str(row["job_id"])
            if key in seen:
                raise ValueError(f"duplicate job_id {key}")
            seen.add(key)
            rows.append(row)
    expected = {str(job["job_id"]) for job in jobs}
    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        raise ValueError(f"score row mismatch: missing={len(missing)} extra={len(extra)}")
    rows.sort(key=lambda row: int(row["job_index"]))
    summary = summarize(
        steps=steps,
        score_rows=rows,
        output_dir=output_dir,
        vote_ks=vote_ks,
        pointwise_summary_path=Path(args.pointwise_summary),
        base_model=args.base_model,
        adapter=args.adapter,
    )
    print(json.dumps({"output_dir": str(output_dir), "gate": summary["stage1_gate"], "best_k": summary["best_k"], "best_accuracy": summary["best_accuracy"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--per-step", default=DEFAULT_PER_STEP)
    parser.add_argument("--pointwise-summary", default=DEFAULT_POINTWISE_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vote-ks", default="8,16,32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vote-chunk", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--score-mode", default="token_prob", choices=["token_prob", "verdict_vote"])
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()

    vote_ks = parse_vote_ks(args.vote_ks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        merge_shards(args, vote_ks)
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
    yes_ids = token_ids_for(processor, [" Yes", "Yes", " yes", "yes"])
    no_ids = token_ids_for(processor, [" No", "No", " no", "no"])
    print(f"yes_ids={yes_ids} no_ids={no_ids}", flush=True)

    steps = read_jsonl(Path(args.per_step))
    if args.limit_steps > 0:
        steps = steps[: args.limit_steps]
    all_jobs, _ = build_jobs(steps)
    indexed_jobs = [(idx, job) for idx, job in enumerate(all_jobs) if idx % args.num_shards == args.shard_index]
    score_path = output_dir / "stage1_candidate_scores.jsonl"
    if args.num_shards > 1:
        score_path = output_dir / f"stage1_candidate_scores.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.jsonl"
    score_rows: List[Dict[str, Any]] = []
    if args.resume and score_path.exists():
        max_k = max(vote_ks)
        for row in read_jsonl(score_path):
            if len(row.get("yes_probs") or []) >= max_k:
                score_rows.append(row)
        seen = {str(row["job_id"]) for row in score_rows}
        indexed_jobs = [(idx, job) for idx, job in indexed_jobs if str(job["job_id"]) not in seen]
        print(f"resumed {len(score_rows)} complete scores from {score_path}", flush=True)
    jobs = [job for _, job in indexed_jobs]
    job_indices = [idx for idx, _ in indexed_jobs]
    print(f"shard {args.shard_index}/{args.num_shards}: scoring {len(jobs)} remaining distinct candidates from {len(all_jobs)} total", flush=True)
    rows = score_rows_for_jobs(
        model=model,
        processor=processor,
        jobs=jobs,
        job_indices=job_indices,
        score_rows=score_rows,
        score_path=score_path,
        vote_ks=vote_ks,
        batch_size=max(1, args.batch_size),
        vote_chunk=max(1, args.vote_chunk),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        yes_ids=yes_ids,
        no_ids=no_ids,
        device=device,
        score_mode=args.score_mode,
    )
    if args.num_shards > 1:
        print(json.dumps({"score_path": str(score_path), "n_distinct_candidates": len(rows)}, indent=2), flush=True)
        return
    summary = summarize(
        steps=steps,
        score_rows=rows,
        output_dir=output_dir,
        vote_ks=vote_ks,
        pointwise_summary_path=Path(args.pointwise_summary),
        base_model=args.base_model,
        adapter=args.adapter,
    )
    print(json.dumps({"output_dir": str(output_dir), "gate": summary["stage1_gate"], "best_k": summary["best_k"], "best_accuracy": summary["best_accuracy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()