#!/usr/bin/env python3
"""Score Stage-2 comparative verifier with a seeded tournament over distinct candidates."""

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

from scripts.build_critstep_verifier_stage2_comparative_data import comparative_prompt  # noqa: E402
from scripts.critstep_verifier_full_action import ORACLE_CRITICAL_TSR_CEILING_PP, projected_tsr_lift_pp  # noqa: E402


DEFAULT_BASE_MODEL = "outputs/critstep_verifier_v2/gui360_fullparam_sft_step250_trainview"
DEFAULT_ADAPTER = "outputs/critstep_verifier_v2/stage2_comparative_lora"
DEFAULT_PER_STEP = "outputs/critstep_verifier_v2/stage1_eval_200_k8_verdict/stage1_per_step.jsonl"
DEFAULT_STAGE1_SUMMARY = "outputs/critstep_verifier_v2/stage1_eval_200_k8_verdict/stage1_summary.json"
DEFAULT_POINTWISE_SUMMARY = "outputs/critstep_verifier/eval_overnight/summary.json"
DEFAULT_OUTPUT_DIR = "outputs/critstep_verifier_v2/stage2_eval_200_tournament"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def render_history(history: Any) -> str:
    if not history:
        return "None"
    if isinstance(history, list):
        return "\n".join(str(item) for item in history) if history else "None"
    return str(history)


def render_control(control: Mapping[str, Any]) -> str:
    if not control or not control.get("key"):
        return "No UIA control metadata."
    return "\n".join([
        f"assignment: {control.get('assignment')}",
        f"label: {control.get('label')}",
        f"control_type: {control.get('type')}",
        f"control_text: {control.get('text')}",
        f"control_rect: {control.get('rect')}",
        f"distance_px: {control.get('distance_px')}",
    ])


def prompt_without_image_marker(prompt: str) -> str:
    if prompt.startswith("<image>\n"):
        return prompt[len("<image>\n") :]
    return prompt.replace("<image>", "", 1).lstrip()


def chat_text(processor: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_without_image_marker(prompt)}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_winner(text: str) -> Optional[str]:
    match = re.search(r"WINNER\s*:\s*([AB])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def candidate_distinct_key(candidate: Mapping[str, Any]) -> str:
    if candidate.get("stage1_distinct_key"):
        return str(candidate["stage1_distinct_key"])
    control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
    payload = {
        "action_signature": candidate.get("action_signature"),
        "pred_type": candidate.get("pred_type"),
        "control_key": control.get("key"),
        "control_assignment": control.get("assignment"),
        "control_rect": control.get("rect"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def candidate_payload(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
    return {
        "action": candidate.get("action") if isinstance(candidate.get("action"), dict) else {},
        "control_text": render_control(control),
    }


def distinct_candidates(step: Mapping[str, Any]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for candidate in step.get("candidates", []):
        key = candidate_distinct_key(candidate)
        score = candidate.get("stage1_score_k8")
        score_value = float(score) if score is not None else 0.0
        current = by_key.get(key)
        if current is None:
            by_key[key] = {
                "distinct_key": key,
                "representative_candidate_id": str(candidate.get("candidate_id")),
                "candidate_ids": [str(candidate.get("candidate_id"))],
                "sources": [candidate.get("source")],
                "is_correct": bool(candidate.get("is_correct")),
                "stage1_score_k8": score_value,
                "action": candidate.get("action") if isinstance(candidate.get("action"), dict) else {},
                "control": candidate.get("control") if isinstance(candidate.get("control"), dict) else {},
                "pred_type": candidate.get("pred_type"),
                "pred_category": candidate.get("pred_category"),
                "bucket": candidate.get("bucket"),
                "reward": candidate.get("reward"),
            }
        else:
            current["candidate_ids"].append(str(candidate.get("candidate_id")))
            current["sources"].append(candidate.get("source"))
            current["is_correct"] = bool(current.get("is_correct") or candidate.get("is_correct"))
            current["stage1_score_k8"] = max(float(current.get("stage1_score_k8") or 0.0), score_value)
            if candidate.get("is_correct") and not current.get("representative_is_correct"):
                current["representative_candidate_id"] = str(candidate.get("candidate_id"))
                current["action"] = candidate.get("action") if isinstance(candidate.get("action"), dict) else {}
                current["control"] = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
                current["representative_is_correct"] = True
    return sorted(by_key.values(), key=lambda item: (-float(item.get("stage1_score_k8") or 0.0), str(item.get("representative_candidate_id"))))


def make_prompt(step: Mapping[str, Any], cand_a: Mapping[str, Any], cand_b: Mapping[str, Any]) -> str:
    return comparative_prompt(
        str(step.get("instruction") or ""),
        render_history(step.get("history")),
        candidate_payload(cand_a),
        candidate_payload(cand_b),
    )


def generate_winners(
    *,
    model: Any,
    processor: Any,
    prompts: Sequence[str],
    image_paths: Sequence[str],
    max_new_tokens: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    tokenizer = processor.tokenizer
    texts = [chat_text(processor, prompt) for prompt in prompts]
    images = [load_image(path) for path in image_paths]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    out: List[Dict[str, Any]] = []
    for sequence in outputs.detach().cpu():
        generated_ids = [int(token_id) for token_id in sequence[input_len:].tolist()]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        out.append({"winner": parse_winner(text), "text_preview": text[:500]})
    return out


def compare_pair(
    *,
    model: Any,
    processor: Any,
    step: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    challenger: Mapping[str, Any],
    max_new_tokens: int,
    device: torch.device,
    bidirectional: bool,
) -> Dict[str, Any]:
    prompts = [make_prompt(step, incumbent, challenger)]
    image_paths = [str(step["screenshot"])]
    orientations = [("incumbent", "challenger")]
    if bidirectional:
        prompts.append(make_prompt(step, challenger, incumbent))
        image_paths.append(str(step["screenshot"]))
        orientations.append(("challenger", "incumbent"))
    generations = generate_winners(
        model=model,
        processor=processor,
        prompts=prompts,
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    votes = {"incumbent": 0, "challenger": 0, "unknown": 0}
    for orientation, generation in zip(orientations, generations, strict=True):
        winner = generation.get("winner")
        if winner == "A":
            votes[orientation[0]] += 1
        elif winner == "B":
            votes[orientation[1]] += 1
        else:
            votes["unknown"] += 1
    if votes["challenger"] > votes["incumbent"]:
        selected = challenger
        selected_role = "challenger"
    elif votes["incumbent"] > votes["challenger"]:
        selected = incumbent
        selected_role = "incumbent"
    else:
        inc_score = float(incumbent.get("stage1_score_k8") or 0.0)
        chal_score = float(challenger.get("stage1_score_k8") or 0.0)
        selected = challenger if chal_score > inc_score else incumbent
        selected_role = "challenger_tie_break" if selected is challenger else "incumbent_tie_break"
    return {
        "incumbent_key": incumbent["distinct_key"],
        "challenger_key": challenger["distinct_key"],
        "incumbent_candidate_id": incumbent["representative_candidate_id"],
        "challenger_candidate_id": challenger["representative_candidate_id"],
        "selected_key": selected["distinct_key"],
        "selected_candidate_id": selected["representative_candidate_id"],
        "selected_role": selected_role,
        "votes": votes,
        "generations": generations,
    }


def run_tournament(
    *,
    model: Any,
    processor: Any,
    step: Mapping[str, Any],
    max_new_tokens: int,
    device: torch.device,
    bidirectional: bool,
) -> Dict[str, Any]:
    candidates = distinct_candidates(step)
    if not candidates:
        raise ValueError(f"no candidates for {step.get('target_id')}")
    incumbent = candidates[0]
    matches = []
    for challenger in candidates[1:]:
        result = compare_pair(
            model=model,
            processor=processor,
            step=step,
            incumbent=incumbent,
            challenger=challenger,
            max_new_tokens=max_new_tokens,
            device=device,
            bidirectional=bidirectional,
        )
        matches.append(result)
        if result["selected_key"] == challenger["distinct_key"]:
            incumbent = challenger
    greedy_key = next((cand["distinct_key"] for cand in candidates if "greedy" in set(cand.get("sources") or [])), None)
    return {
        "target_id": step["target_id"],
        "episode_id": step.get("episode_id"),
        "step_idx": step.get("step_idx"),
        "subset": step.get("subset"),
        "depth_bin": step.get("depth_bin"),
        "n_distinct_candidates": len(candidates),
        "seed_candidate_id": candidates[0]["representative_candidate_id"],
        "seed_correct": bool(candidates[0].get("is_correct")),
        "stage2_candidate_id": incumbent["representative_candidate_id"],
        "stage2_distinct_key": incumbent["distinct_key"],
        "stage2_score_seed": incumbent.get("stage1_score_k8"),
        "stage2_correct": bool(incumbent.get("is_correct")),
        "stage2_reject_greedy": bool(greedy_key is not None and incumbent["distinct_key"] != greedy_key),
        "matches": matches,
    }


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


def stage2_gate(summary: Mapping[str, Any]) -> str:
    stage2_acc = summary["selection_accuracy"].get("stage2_tournament") or 0.0
    stage1_acc = summary["selection_accuracy"].get("stage1_cot_vote_k8") or 0.0
    stage2_deep = summary.get("depth_stratified", {}).get("deep_21_50", {}).get("stage2_correct") or 0.0
    stage1_deep = summary.get("stage1_baseline", {}).get("deep_21_50") or 0.0
    stage2_elem = summary.get("subset_stratified", {}).get("click_element_selection", {}).get("stage2_correct") or 0.0
    stage1_elem = summary.get("stage1_baseline", {}).get("click_element_selection") or 0.0
    if stage2_acc > stage1_acc + 0.02 and (stage2_deep > stage1_deep + 0.02 or stage2_elem > stage1_elem + 0.02):
        return "COMPARATIVE EFFECTIVE"
    return "COMPARATIVE PLATEAU"


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Stage 2 Comparative Ranking Verifier", ""]
    lines.append("## Selection Accuracy")
    lines.append("")
    lines.append("| selector | accuracy | projected TSR lift proxy | fraction of +23.77pp ceiling |")
    lines.append("|---|---:|---:|---:|")
    for name, accuracy in summary["selection_accuracy"].items():
        lift = projected_tsr_lift_pp(accuracy, summary["recoverable_fraction_primary"]) if accuracy is not None else None
        frac = lift / ORACLE_CRITICAL_TSR_CEILING_PP if lift is not None else None
        if accuracy is None:
            lines.append(f"| {name} | NA | NA | NA |")
        else:
            lines.append(f"| {name} | {accuracy*100:.2f}% | {lift:.2f}pp | {frac*100:.2f}% |")
    lines.append("")
    lines.append("## Depth-Stratified")
    lines.append("")
    lines.append("| depth bin | n | Stage2 tournament | Stage1 CoT K=8 | Pointwise |")
    lines.append("|---|---:|---:|---:|---:|")
    stage1 = summary.get("stage1_baseline", {})
    pointwise = summary.get("pointwise_baseline", {})
    for depth, item in summary["depth_stratified"].items():
        s1 = stage1.get(depth)
        pw = pointwise.get(depth)
        s1_text = f"{s1*100:.2f}%" if s1 is not None else "NA"
        pw_text = f"{pw*100:.2f}%" if pw is not None else "NA"
        lines.append(f"| {depth} | {item['n']} | {item['stage2_correct']*100:.2f}% | {s1_text} | {pw_text} |")
    lines.append("")
    lines.append("## Per-Subset")
    lines.append("")
    lines.append("| subset | n | Stage2 tournament | Stage1 CoT K=8 |")
    lines.append("|---|---:|---:|---:|")
    for subset, item in summary["subset_stratified"].items():
        s1 = stage1.get(subset)
        s1_text = f"{s1*100:.2f}%" if s1 is not None else "NA"
        lines.append(f"| {subset} | {item['n']} | {item['stage2_correct']*100:.2f}% | {s1_text} |")
    lines.append("")
    lines.append("## Reject-Greedy")
    lines.append("")
    lines.append(f"Reject-greedy rate: `{summary['reject_greedy_rate']*100:.2f}%`")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['stage2_gate']}**")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'stage2_comparative.md'}`")
    lines.append(f"- `{output_dir / 'stage2_summary.json'}`")
    lines.append(f"- `{output_dir / 'stage2_per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def summarize(rows: List[Dict[str, Any]], output_dir: Path, stage1_summary_path: Path, pointwise_summary_path: Path, base_model: str, adapter: str) -> Dict[str, Any]:
    stage1 = load_json(stage1_summary_path, {}) or {}
    pointwise = load_json(pointwise_summary_path, {}) or {}
    stage1_depth = stage1.get("best_depth_accuracy") or {}
    stage1_subset = {key: value.get("stage1_k8_correct") for key, value in (stage1.get("subset_stratified_best") or {}).items()}
    pointwise_depth = {key: value.get("verifier_correct") for key, value in (pointwise.get("depth_stratified") or {}).items()}
    selection_accuracy: Dict[str, Optional[float]] = {
        "oracle_in_pool": 1.0,
        "greedy": fraction(rows, "greedy_correct"),
        "sample_order_first": fraction(rows, "first_sample_correct"),
        "previous_pointwise_verifier": (pointwise.get("selection_accuracy") or {}).get("verifier_argmax"),
        "stage1_cot_vote_k8": (stage1.get("selection_accuracy") or {}).get("cot_vote_k8"),
        "stage2_tournament": fraction(rows, "stage2_correct"),
    }
    recoverable_fraction = 488 / 862
    stage2_acc = selection_accuracy["stage2_tournament"] or 0.0
    lift = projected_tsr_lift_pp(stage2_acc, recoverable_fraction) or 0.0
    summary = {
        "base_model": base_model,
        "adapter": adapter,
        "aggregation": "stage1_seeded_tournament_bidirectional",
        "n_steps": len(rows),
        "n_distinct_candidates": sum(int(row.get("n_distinct_candidates") or 0) for row in rows),
        "n_matches": sum(len(row.get("matches") or []) for row in rows),
        "selection_accuracy": selection_accuracy,
        "stage1_baseline": {"overall": selection_accuracy.get("stage1_cot_vote_k8"), **stage1_depth, **stage1_subset},
        "pointwise_baseline": {"overall": selection_accuracy.get("previous_pointwise_verifier"), **pointwise_depth},
        "depth_stratified": summarize_by(rows, "depth_bin", ["stage2_correct"]),
        "subset_stratified": summarize_by(rows, "subset", ["stage2_correct"]),
        "reject_greedy_rate": fraction(rows, "stage2_reject_greedy"),
        "n_primary_failures": 862,
        "n_recoverable_primary": 488,
        "recoverable_fraction_primary": recoverable_fraction,
        "projected_tsr_lift_pp": lift,
        "projected_ceiling_fraction": lift / ORACLE_CRITICAL_TSR_CEILING_PP,
    }
    summary["stage2_gate"] = stage2_gate(summary)
    write_json(output_dir / "stage2_summary.json", summary)
    (output_dir / "stage2_comparative.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    return summary


def merge_shards(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    shard_paths = sorted(output_dir.glob(f"stage2_per_step.shard_*_of_{args.num_shards:02d}.jsonl"))
    if len(shard_paths) != args.num_shards:
        raise FileNotFoundError(f"expected {args.num_shards} shard files, found {len(shard_paths)}")
    rows: List[Dict[str, Any]] = []
    seen = set()
    for path in shard_paths:
        for row in read_jsonl(path):
            key = str(row["target_id"])
            if key in seen:
                raise ValueError(f"duplicate target_id {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: int(row.get("step_global_index", 10**12)))
    expected = len(read_jsonl(Path(args.per_step)))
    if args.limit_steps > 0:
        expected = min(expected, args.limit_steps)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} scored steps, found {len(rows)}")
    write_jsonl(output_dir / "stage2_per_step.jsonl", rows)
    summary = summarize(rows, output_dir, Path(args.stage1_summary), Path(args.pointwise_summary), args.base_model, args.adapter)
    print(json.dumps({"output_dir": str(output_dir), "gate": summary["stage2_gate"], "stage2_accuracy": summary["selection_accuracy"]["stage2_tournament"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--per-step", default=DEFAULT_PER_STEP)
    parser.add_argument("--stage1-summary", default=DEFAULT_STAGE1_SUMMARY)
    parser.add_argument("--pointwise-summary", default=DEFAULT_POINTWISE_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)  # reserved for future batched generation
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--no-bidirectional", dest="bidirectional", action="store_false")
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
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

    steps = read_jsonl(Path(args.per_step))
    if args.limit_steps > 0:
        steps = steps[: args.limit_steps]
    indexed_steps = [(idx, step) for idx, step in enumerate(steps) if idx % args.num_shards == args.shard_index]
    score_path = output_dir / "stage2_per_step.jsonl"
    if args.num_shards > 1:
        score_path = output_dir / f"stage2_per_step.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.jsonl"

    scored_rows: List[Dict[str, Any]] = []
    seen = set()
    if args.resume and score_path.exists():
        scored_rows = read_jsonl(score_path)
        seen = {str(row["target_id"]) for row in scored_rows}
        indexed_steps = [(idx, step) for idx, step in indexed_steps if str(step["target_id"]) not in seen]
        print(f"resumed {len(scored_rows)} scored steps from {score_path}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"loading processor/model base={args.base_model} adapter={args.adapter} device={device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval().to(device)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"shard {args.shard_index}/{args.num_shards}: scoring {len(indexed_steps)} remaining steps from {len(steps)} total", flush=True)

    for local_idx, (step_idx, step) in enumerate(indexed_steps, start=1):
        row = run_tournament(
            model=model,
            processor=processor,
            step=step,
            max_new_tokens=args.max_new_tokens,
            device=device,
            bidirectional=args.bidirectional,
        )
        row["step_global_index"] = step_idx
        row["greedy_correct"] = step.get("greedy_correct")
        row["first_sample_correct"] = step.get("first_sample_correct")
        scored_rows.append(row)
        write_jsonl(score_path, scored_rows)
        print(f"scored {local_idx}/{len(indexed_steps)} steps; target={row['target_id']} correct={row['stage2_correct']}", flush=True)
    if args.num_shards > 1:
        print(json.dumps({"score_path": str(score_path), "n_steps": len(scored_rows)}, indent=2), flush=True)
        return
    summary = summarize(scored_rows, output_dir, Path(args.stage1_summary), Path(args.pointwise_summary), args.base_model, args.adapter)
    print(json.dumps({"output_dir": str(output_dir), "gate": summary["stage2_gate"], "stage2_accuracy": summary["selection_accuracy"]["stage2_tournament"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()