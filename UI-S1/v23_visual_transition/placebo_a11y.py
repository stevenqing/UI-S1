#!/usr/bin/env python3
"""Placebo-a11y mechanism test for GUI-360.

Separates information effect from prompting/re-examination effect by running
scrambled but format-matched placebo a11y prompts on the Phase 1 expanded slice.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from openai import OpenAI

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v23_visual_transition.modality_jaccard import (  # noqa: E402
    VA_PROMPT,
    action_signature,
    annotate_mechanisms,
    attach_controls,
    bootstrap_ci,
    classify_prediction,
    compact_controls,
    encode_image,
    ranked_controls,
    read_balanced_states,
)
from v23_visual_transition.va_symbolic import (  # noqa: E402
    GROUNDING_BUCKETS,
    SYMBOLIC_PROMPT,
    symbolic_annotations,
)
from v13_gui_360.eval_gui360_template import SUPPORTED_ACTIONS  # noqa: E402

BUCKETS = ("far_miss", "type_mismatch")
REAL_SOURCES = ("VA", "symbolic")
PLACEBO_SOURCES = ("placebo_full", "placebo_symbolic")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_rect(ctrl: Dict[str, Any]) -> Any:
    return ctrl.get("control_rect") or ctrl.get("bbox") or ctrl.get("rectangle") or []


def derange_controls(controls: Sequence[Dict[str, Any]], seed_key: str) -> List[Dict[str, Any]]:
    controls = [ctrl for ctrl in controls if isinstance(ctrl, dict)]
    n = len(controls)
    if n == 0:
        return []
    if n == 1:
        ctrl = copy.deepcopy(controls[0])
        ctrl["control_type"] = "PlaceboControl"
        ctrl["control_text"] = "placebo element"
        ctrl["label"] = -1
        rect = get_rect(ctrl)
        if isinstance(rect, list) and len(rect) >= 4:
            ctrl["control_rect"] = [rect[0] + 37, rect[1] + 29, rect[2] + 37, rect[3] + 29]
        return [ctrl]
    rng = random.Random(seed_key)
    type_perm = list(range(n))
    text_perm = list(range(n))
    rect_perm = list(range(n))
    for perm in (type_perm, text_perm, rect_perm):
        rng.shuffle(perm)
        for idx in range(n):
            if perm[idx] == idx:
                swap = (idx + 1) % n
                perm[idx], perm[swap] = perm[swap], perm[idx]
    out: List[Dict[str, Any]] = []
    for idx, ctrl in enumerate(controls):
        type_src = controls[type_perm[idx]]
        text_src = controls[text_perm[idx]]
        rect_src = controls[rect_perm[idx]]
        new_ctrl = copy.deepcopy(ctrl)
        new_ctrl["control_type"] = type_src.get("control_type") or type_src.get("type") or "Unknown"
        new_ctrl["control_text"] = text_src.get("control_text") or text_src.get("name") or text_src.get("text") or ""
        new_ctrl["label"] = text_src.get("label", idx + 1)
        rect = get_rect(rect_src)
        if rect:
            new_ctrl["control_rect"] = copy.deepcopy(rect)
        for key in ("bbox", "rectangle"):
            if key in new_ctrl and rect:
                new_ctrl[key] = copy.deepcopy(rect)
        out.append(new_ctrl)
    return out


def history_text(state: Dict[str, Any]) -> str:
    return "\n".join(state.get("history") or []) if state.get("history") else "None"


def build_placebo_full_messages(state: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    real_controls = ranked_controls(state, args.max_full_controls)
    placebo_controls = derange_controls(real_controls, f"{args.seed}:{state['state_id']}:full")
    elements = compact_controls(placebo_controls, args.max_full_controls)
    image_url = f"data:image/png;base64,{encode_image(state['image_bytes'], args.image_max_pixels)}"
    text = VA_PROMPT.format(
        instruction=state["goal"],
        history=history_text(state),
        elements=elements,
        actions=SUPPORTED_ACTIONS,
    )
    meta = {
        "source": "placebo_full",
        "format_matched_to": "unconditional_full_a11y",
        "controls_total": len(state.get("controls") or []),
        "controls_serialized": min(len(real_controls), args.max_full_controls),
        "same_count_as_real": True,
        "scrambled_fields": ["control_type", "label", "control_text", "control_rect"],
        "contains_control_rect": True,
        "contains_coordinates": True,
    }
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": text}]}], meta


def build_placebo_symbolic_messages(state: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    real_controls = list(state.get("controls") or [])[: args.max_symbolic_controls]
    placebo_controls = derange_controls(real_controls, f"{args.seed}:{state['state_id']}:symbolic")
    annotations, base_meta = symbolic_annotations(placebo_controls, "symbolic", args.max_symbolic_controls)
    image_url = f"data:image/png;base64,{encode_image(state['image_bytes'], args.image_max_pixels)}"
    text = SYMBOLIC_PROMPT.format(
        instruction=state["goal"],
        history=history_text(state),
        annotations=annotations,
        actions=SUPPORTED_ACTIONS,
    )
    meta = dict(base_meta)
    meta.update({
        "source": "placebo_symbolic",
        "format_matched_to": "symbolic_type+label_no_coords",
        "same_count_as_real": True,
        "scrambled_fields": ["control_type", "label", "control_text"],
        "contains_control_rect": False,
        "contains_coordinates": False,
    })
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": text}]}], meta


def evaluate_one(args: argparse.Namespace, state: Dict[str, Any], real_row: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    row: Dict[str, Any] = {
        "state_id": state["state_id"],
        "episode_id": state.get("episode_id"),
        "step_idx": state.get("step_idx"),
        "goal": state.get("goal"),
        "gt_action": state.get("gt_action"),
        "V_bucket": real_row["V"].get("bucket"),
        "a11y_present": bool(state.get("controls")),
        "num_controls": len(state.get("controls") or []),
        "V": real_row["V"],
        "VA": real_row["VA"],
        "symbolic": real_row["symbolic"],
    }
    builders = {
        "placebo_full": build_placebo_full_messages,
        "placebo_symbolic": build_placebo_symbolic_messages,
    }
    for source, builder in builders.items():
        messages, meta = builder(state, args)
        row[f"{source}_serialization"] = meta
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            pred_text = response.choices[0].message.content or ""
        except Exception as exc:
            pred_text = ""
            row[f"{source}_api_error"] = str(exc)[:240]
        row[source] = classify_prediction(
            pred_text,
            state["gt_action"],
            state["image_w"],
            state["image_h"],
            args.match_threshold,
            args.near_px,
            args.far_px,
        )
        row[f"agreement_V_{source}"] = action_signature(row["V"]) == action_signature(row[source])
    return row


def evaluate_states(args: argparse.Namespace, states: Sequence[Dict[str, Any]], real_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if args.threads <= 1:
        rows = []
        for idx, state in enumerate(states, 1):
            rows.append(evaluate_one(args, state, real_by_id[state["state_id"]]))
            if args.log_every and idx % args.log_every == 0:
                print(f"evaluated {idx}/{len(states)} states", flush=True)
        return rows
    rows = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(evaluate_one, args, state, real_by_id[state["state_id"]]) for state in states]
        for idx, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if args.log_every and idx % args.log_every == 0:
                print(f"evaluated {idx}/{len(states)} states", flush=True)
    rows.sort(key=lambda item: (item.get("episode_id", ""), int(item.get("step_idx", 0)), item["state_id"]))
    return rows


def mean_ci(values: Sequence[float], seed: int, samples: int) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(samples, len(arr)))
    boot = arr[idx].mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def repair(rows: Sequence[Dict[str, Any]], source: str, bucket: str, args: argparse.Namespace) -> Dict[str, Any]:
    sub = [row for row in rows if row["V_bucket"] == bucket]
    values = [float(row[source]["success"]) - float(row["V"]["success"]) for row in sub]
    return {
        "n": len(sub),
        "correct_rate": sum(float(row[source]["success"]) for row in sub) / max(len(sub), 1),
        "repair_vs_V": mean_ci(values, args.seed, args.bootstrap_samples),
    }


def paired_diff(rows: Sequence[Dict[str, Any]], real: str, placebo: str, bucket: str, args: argparse.Namespace) -> Dict[str, Any]:
    sub = [row for row in rows if row["V_bucket"] == bucket]
    values = [float(row[real]["success"]) - float(row[placebo]["success"]) for row in sub]
    return {"n": len(sub), "diff": mean_ci(values, args.seed, args.bootstrap_samples)}


def source_sig(contrast_by_bucket: Dict[str, Dict[str, Any]]) -> bool:
    return any(item["diff"][1] > 0.0 for item in contrast_by_bucket.values())


def source_zeroish(contrast_by_bucket: Dict[str, Dict[str, Any]], tolerance: float) -> bool:
    return all(abs(item["diff"][0]) <= tolerance and item["diff"][1] <= 0.0 <= item["diff"][2] for item in contrast_by_bucket.values())


def summarize(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    repair_table = {bucket: {} for bucket in BUCKETS}
    for bucket in BUCKETS:
        for source in ["symbolic", "placebo_symbolic", "VA", "placebo_full"]:
            repair_table[bucket][source] = repair(rows, source, bucket, args)
    d1 = {bucket: paired_diff(rows, "symbolic", "placebo_symbolic", bucket, args) for bucket in BUCKETS}
    d2 = {bucket: paired_diff(rows, "VA", "placebo_full", bucket, args) for bucket in BUCKETS}
    support_ok = all(repair_table[bucket]["VA"]["n"] >= 30 for bucket in BUCKETS)
    d1_sig = source_sig(d1)
    d2_sig = source_sig(d2)
    d1_zero = source_zeroish(d1, args.zero_tolerance)
    d2_zero = source_zeroish(d2, args.zero_tolerance)
    if not support_ok:
        verdict = "PENDING_SUPPORT"
        consequent = "expanded slice lacks n>=30 in a bucket; do not decide modality line"
    elif d1_sig and d2_sig:
        verdict = "INFORMATION_EFFECT"
        consequent = "real a11y content beats placebo in both real conditions; reopen conditional full-a11y + verifier"
    elif d1_sig or d2_sig:
        verdict = "MIXED"
        consequent = "partial information effect; scope modality source to the channel/failure type where real beats placebo"
    elif d1_zero and d2_zero:
        verdict = "PROMPTING_EFFECT"
        consequent = "placebo repairs about as well as real; modality line is prompt/re-examination effect, converge to single-agent"
    else:
        verdict = "INCONCLUSIVE"
        consequent = "no significant real-content advantage; not enough for INFORMATION_EFFECT, but not pre-registered zero-ish either"
    return {
        "n": len(rows),
        "support_ok": support_ok,
        "repair_table": repair_table,
        "D1_real_symbolic_minus_placebo_symbolic": d1,
        "D2_real_full_minus_placebo_full": d2,
        "d1_significant": d1_sig,
        "d2_significant": d2_sig,
        "d1_zeroish": d1_zero,
        "d2_zeroish": d2_zero,
        "verdict": verdict,
        "consequent": consequent,
        "api_errors": {source: sum(1 for row in rows if row.get(f"{source}_api_error")) for source in PLACEBO_SOURCES},
    }


def fmt_ci(values: Sequence[float]) -> str:
    return f"{values[0]:+.4f} [{values[1]:+.4f}, {values[2]:+.4f}]"


def example_pairs(states: Sequence[Dict[str, Any]], args: argparse.Namespace, count: int = 2) -> List[Dict[str, Any]]:
    examples = []
    for state in states[:count]:
        full_real_controls = ranked_controls(state, args.max_full_controls)
        full_placebo_controls = derange_controls(full_real_controls, f"{args.seed}:{state['state_id']}:full")
        symbolic_real_controls = list(state.get("controls") or [])[: args.max_symbolic_controls]
        symbolic_placebo_controls = derange_controls(symbolic_real_controls, f"{args.seed}:{state['state_id']}:symbolic")
        real_full = compact_controls(full_real_controls[:5], 5).splitlines()
        placebo_full = compact_controls(full_placebo_controls[:5], 5).splitlines()
        real_sym, _ = symbolic_annotations(symbolic_real_controls[:5], "symbolic", 5)
        placebo_sym, _ = symbolic_annotations(symbolic_placebo_controls[:5], "symbolic", 5)
        examples.append({
            "state_id": state["state_id"],
            "real_full_first5": real_full,
            "placebo_full_first5": placebo_full,
            "real_symbolic_first5": real_sym.splitlines(),
            "placebo_symbolic_first5": placebo_sym.splitlines(),
        })
    return examples


def render(summary: Dict[str, Any], examples: Sequence[Dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# Placebo-A11y: Information Effect vs Prompting Effect",
        "",
        "## Gate Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        summary["consequent"],
        "",
        "## Inputs",
        "",
        f"- modality rows: `{args.modality_rows}`",
        f"- symbolic rows: `{args.symbolic_rows}`",
        f"- joined/evaluated states: `{summary['n']}`",
        "- split: `GUI-360 balanced test`",
        "- buckets: frozen V trichotomy buckets",
        "- placebo keeps same prompt framing, element count, and serialization format; only content is scrambled",
        f"- placebo API errors: `{summary['api_errors']}`",
        "",
        "## Placebo-vs-Real Examples",
        "",
    ]
    for ex in examples:
        lines += [f"### State `{ex['state_id']}`", "", "real full first 5:", "```text", *ex["real_full_first5"], "```", "placebo full first 5:", "```text", *ex["placebo_full_first5"], "```", "real symbolic first 5:", "```text", *ex["real_symbolic_first5"], "```", "placebo symbolic first 5:", "```text", *ex["placebo_symbolic_first5"], "```", ""]
    lines += [
        "## Repair Table",
        "",
        "| bucket | n | condition | correct | repair vs V |",
        "|---|---:|---|---:|---:|",
    ]
    labels = {
        "symbolic": "real_symbolic",
        "placebo_symbolic": "placebo_symbolic",
        "VA": "real_full_a11y",
        "placebo_full": "placebo_full",
    }
    for bucket in BUCKETS:
        for source in ["symbolic", "placebo_symbolic", "VA", "placebo_full"]:
            row = summary["repair_table"][bucket][source]
            lines.append(f"| {bucket} | {row['n']} | {labels[source]} | {row['correct_rate']:.4f} | {fmt_ci(row['repair_vs_V'])} |")
    lines += [
        "",
        "## Paired Placebo Contrasts",
        "",
        "| contrast | bucket | n | real - placebo | verdict hint |",
        "|---|---|---:|---:|---|",
    ]
    for bucket, item in summary["D1_real_symbolic_minus_placebo_symbolic"].items():
        hint = "significant" if item["diff"][1] > 0.0 else "ci_includes_0"
        lines.append(f"| D1 symbolic content | {bucket} | {item['n']} | {fmt_ci(item['diff'])} | {hint} |")
    for bucket, item in summary["D2_real_full_minus_placebo_full"].items():
        hint = "significant" if item["diff"][1] > 0.0 else "ci_includes_0"
        lines.append(f"| D2 full content | {bucket} | {item['n']} | {fmt_ci(item['diff'])} | {hint} |")
    lines += [
        "",
        "## Decision Flags",
        "",
        f"- support_ok: `{summary['support_ok']}`",
        f"- D1 significant: `{summary['d1_significant']}`",
        f"- D2 significant: `{summary['d2_significant']}`",
        f"- D1 zero-ish: `{summary['d1_zeroish']}`",
        f"- D2 zero-ish: `{summary['d2_zeroish']}`",
        f"- information-effect support: `{summary['d1_significant'] or summary['d2_significant']}`",
        "",
        "## One-Line Consequent",
        "",
        summary["consequent"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced_data_dir", default="datasets/gui360-balanced/data")
    parser.add_argument("--raw_repo", default="vyokky/GUI-360")
    parser.add_argument("--raw_local_dir", default="datasets/GUI-360-raw-jsonl")
    parser.add_argument("--modality_rows", default="outputs/candidate_orthogonality/cond_full_a11y/phase1_modality260/per_state.jsonl")
    parser.add_argument("--symbolic_rows", default="outputs/candidate_orthogonality/cond_full_a11y/phase1_symbolic260/per_state.jsonl")
    parser.add_argument("--output_dir", default="outputs/candidate_orthogonality/placebo_a11y")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--limit", type=int, default=260)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max_full_controls", type=int, default=256)
    parser.add_argument("--max_symbolic_controls", type=int, default=512)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--zero_tolerance", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    modality_rows = read_jsonl(Path(args.modality_rows))
    symbolic_rows = read_jsonl(Path(args.symbolic_rows))
    symbolic_by_id = {row["state_id"]: row for row in symbolic_rows}
    real_by_id = {}
    for row in modality_rows:
        sym = symbolic_by_id.get(row["state_id"])
        if sym is None:
            continue
        real_by_id[row["state_id"]] = {"V": row["V"], "VA": row["VA"], "symbolic": sym["symbolic"]}

    all_states = read_balanced_states(args.balanced_data_dir, 0)
    rng = random.Random(args.seed)
    rng.shuffle(all_states)
    states = all_states[: args.limit]
    attach_controls(states, args.raw_repo, args.raw_local_dir, args.log_every)
    for state in states:
        annotate_mechanisms(state)
    states = [state for state in states if state["state_id"] in real_by_id and state.get("controls")]
    if len(states) != len(real_by_id):
        print(f"Warning: matched {len(states)} states for {len(real_by_id)} real rows", flush=True)

    rows = evaluate_states(args, states, real_by_id)
    summary = summarize(rows, args)
    examples = example_pairs(states, args, count=2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "per_state.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "examples": examples, "args": vars(args)}, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "summary.md").write_text(render(summary, examples, args))
    print(f"Wrote {out_dir / 'summary.md'}")
    print(f"Wrote {out_dir / 'per_state.jsonl'}")
    print(f"PLACEBO_GATE: {summary['verdict']} - {summary['consequent']}")


if __name__ == "__main__":
    main()
