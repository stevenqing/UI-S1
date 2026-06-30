#!/usr/bin/env python3
"""V+A_symbolic modality source gate for GUI-360.

Runs the frozen SFT model with symbolic UIA annotations only: no bboxes and no
coordinates in the injected a11y text. Evaluates full symbolic plus type-only and
label-only ablations against saved visual-only baseline rows.
"""

from __future__ import annotations

import argparse
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

from v13_gui_360.eval_gui360_template import SUPPORTED_ACTIONS  # noqa: E402
from v23_visual_transition.modality_jaccard import (  # noqa: E402
    action_signature,
    annotate_mechanisms,
    attach_controls,
    bootstrap_ci,
    classify_prediction,
    encode_image,
    read_balanced_states,
)

SOURCE_SPECS = {
    "symbolic": {"title": "symbolic type+label", "include_type": True, "include_label": True},
    "type_only": {"title": "type-only", "include_type": True, "include_label": False},
    "label_only": {"title": "label-only", "include_type": False, "include_label": True},
}
GROUNDING_BUCKETS = ("far_miss", "type_mismatch")
UNCONDITIONAL_JACCARD = 0.6145833333333334
PROMPT_JACCARD = 0.91
UNCONDITIONAL_ORACLE = 0.705

SYMBOLIC_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, decide the next action.

The instruction is:
{instruction}

The history of actions are:
{history}

Symbolic accessibility annotations for elements visible on the current screen:
{annotations}

These annotations intentionally contain NO coordinates or bounding boxes. Use the screenshot for location and layout. Use the symbolic annotations only as complementary information: element semantic identity/name and/or affordance/type.

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, reason from the screenshot to locate the target visually. Use the symbolic annotations only to disambiguate what the target is called and what affordance it has. Then output one action within <tool_call></tool_call>:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

Only ONE action should be taken at a time.
"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any, limit: int = 50) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())[:limit]


def symbolic_annotations(controls: Sequence[Dict[str, Any]], source: str, max_controls: int) -> Tuple[str, Dict[str, Any]]:
    spec = SOURCE_SPECS[source]
    lines = []
    truncated = max(0, len(controls) - max_controls)
    for idx, ctrl in enumerate(controls[:max_controls], 1):
        parts = [f"element {idx}"]
        if spec["include_type"]:
            parts.append(f"type={clean_text(ctrl.get('control_type') or ctrl.get('type') or 'Unknown', 36)!r}")
        if spec["include_label"]:
            label = ctrl.get("label")
            text = clean_text(ctrl.get("control_text") or ctrl.get("name") or ctrl.get("text") or "", 50)
            if label is not None:
                parts.append(f"label={label!r}")
            if text:
                parts.append(f"name={text!r}")
        lines.append("; ".join(parts))
    if truncated:
        lines.append(f"... {truncated} additional elements omitted; no coordinates were provided")
    if not lines:
        lines.append("(no symbolic accessibility annotations available)")
    meta = {
        "source": source,
        "controls_total": len(controls),
        "controls_serialized": min(len(controls), max_controls),
        "controls_truncated": truncated,
        "contains_control_rect": False,
        "contains_coordinates": False,
        "fields": [field for field, enabled in (("control_type", spec["include_type"]), ("label/control_text", spec["include_label"])) if enabled],
    }
    return "\n".join(lines), meta


def build_symbolic_messages(state: Dict[str, Any], source: str, max_controls: int, image_max_pixels: int | None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    history = "\n".join(state.get("history") or []) if state.get("history") else "None"
    annotations, meta = symbolic_annotations(state.get("controls") or [], source, max_controls)
    image_url = f"data:image/png;base64,{encode_image(state['image_bytes'], image_max_pixels)}"
    text = SYMBOLIC_PROMPT.format(
        instruction=state["goal"],
        history=history,
        annotations=annotations,
        actions=SUPPORTED_ACTIONS,
    )
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": text}]}], meta


def evaluate_one_state(args: argparse.Namespace, state: Dict[str, Any], v_row: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    row = {
        "state_id": state["state_id"],
        "episode_id": state["episode_id"],
        "step_idx": state["step_idx"],
        "goal": state["goal"],
        "screenshot": state.get("screenshot"),
        "gt_action": state["gt_action"],
        "a11y_present": bool(state.get("controls")),
        "a11y_sparse": len(state.get("controls") or []) < 10,
        "num_controls": len(state.get("controls") or []),
        "V": v_row["V"],
        "V_bucket": v_row["V"].get("bucket"),
        "failure_type_prediction": {
            "source": "frozen_matcher_V_bucket",
            "bucket": v_row["V"].get("bucket"),
            "grounding_prone": v_row["V"].get("bucket") in GROUNDING_BUCKETS,
        },
    }
    for source in ["symbolic", "type_only", "label_only"]:
        messages, meta = build_symbolic_messages(state, source, args.max_controls, args.image_max_pixels)
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
    row["verifier_choice"] = None
    row["final_correct"] = None
    return row


def evaluate_states(args: argparse.Namespace, states: Sequence[Dict[str, Any]], v_rows_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if args.threads <= 1:
        for idx, state in enumerate(states, 1):
            rows.append(evaluate_one_state(args, state, v_rows_by_id[state["state_id"]]))
            if args.log_every and idx % args.log_every == 0:
                print(f"evaluated {idx}/{len(states)} states", flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(evaluate_one_state, args, state, v_rows_by_id[state["state_id"]]) for state in states]
        for idx, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if args.log_every and idx % args.log_every == 0:
                print(f"evaluated {idx}/{len(states)} states", flush=True)
    rows.sort(key=lambda item: (item["episode_id"], int(item["step_idx"]), item["state_id"]))
    return rows


def mean_ci(values: Sequence[float], seed: int) -> Tuple[float, float, float]:
    return bootstrap_ci(values, seed)


def jaccard(rows: Sequence[Dict[str, Any]], source: str) -> float:
    v_errors = {row["state_id"] for row in rows if not row["V"]["success"]}
    s_errors = {row["state_id"] for row in rows if not row[source]["success"]}
    union = v_errors | s_errors
    return len(v_errors & s_errors) / max(len(union), 1)


def bootstrap_jaccard(rows: Sequence[Dict[str, Any]], source: str, seed: int, samples: int = 5000) -> Tuple[float, float, float]:
    if not rows:
        return 0.0, 0.0, 0.0
    point = jaccard(rows, source)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(rows)
    for _ in range(samples):
        vals.append(jaccard([rows[i] for i in rng.integers(0, n, size=n)], source))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def source_summary(rows: Sequence[Dict[str, Any]], source: str, seed: int) -> Dict[str, Any]:
    n = len(rows)
    v_correct = [float(row["V"]["success"]) for row in rows]
    s_correct = [float(row[source]["success"]) for row in rows]
    delta = [s - v for v, s in zip(v_correct, s_correct)]
    cells = Counter()
    for row in rows:
        v_ok = row["V"]["success"]
        s_ok = row[source]["success"]
        if v_ok and s_ok:
            cells["both_right"] += 1
        elif v_ok and not s_ok:
            cells["only_V_right"] += 1
        elif s_ok and not v_ok:
            cells[f"only_{source}_right"] += 1
        else:
            cells["neither_right"] += 1
    axis = {}
    for bucket in GROUNDING_BUCKETS:
        sub = [row for row in rows if row["V_bucket"] == bucket]
        values = [float(row[source]["success"]) - float(row["V"]["success"]) for row in sub]
        axis[bucket] = {
            "n": len(sub),
            "V_correct_rate": sum(float(row["V"]["success"]) for row in sub) / max(len(sub), 1),
            "source_correct_rate": sum(float(row[source]["success"]) for row in sub) / max(len(sub), 1),
            "source_minus_V": mean_ci(values, seed),
            "support_ok": len(sub) >= 30,
        }
    disagreement = [row for row in rows if not row[f"agreement_V_{source}"]]
    return {
        "source": source,
        "n": n,
        "api_errors": sum(1 for row in rows if row.get(f"{source}_api_error")),
        "V_correct_rate": sum(v_correct) / max(n, 1),
        "source_correct_rate": sum(s_correct) / max(n, 1),
        "source_minus_V": mean_ci(delta, seed),
        "axis": axis,
        "jaccard": bootstrap_jaccard(rows, source, seed),
        "jaccard_disagreement_subset": bootstrap_jaccard(disagreement, source, seed) if disagreement else (0.0, 0.0, 0.0),
        "agreement": mean_ci([float(row[f"agreement_V_{source}"]) for row in rows], seed),
        "unique_coverage": dict(cells),
        "oracle_ceiling": sum(1 for row in rows if row["V"]["success"] or row[source]["success"]) / max(n, 1),
        "disagreement_n": len(disagreement),
    }


def paired_delta(rows: Sequence[Dict[str, Any]], bucket: str, source_a: str, source_b: str, seed: int) -> Tuple[float, float, float, int]:
    sub = [row for row in rows if row["V_bucket"] == bucket]
    values = [float(row[source_a]["success"]) - float(row[source_b]["success"]) for row in sub]
    mean, lo, hi = mean_ci(values, seed)
    return mean, lo, hi, len(sub)


def summarize(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    summaries = {source: source_summary(rows, source, args.seed) for source in SOURCE_SPECS}
    dissociation = {
        "P3a_label_to_far_miss": {
            "bucket": "far_miss",
            "full_minus_type_only": paired_delta(rows, "far_miss", "symbolic", "type_only", args.seed),
            "full_gain": summaries["symbolic"]["axis"]["far_miss"]["source_minus_V"],
            "type_only_gain": summaries["type_only"]["axis"]["far_miss"]["source_minus_V"],
            "label_only_gain": summaries["label_only"]["axis"]["far_miss"]["source_minus_V"],
        },
        "P3b_type_to_type_mismatch": {
            "bucket": "type_mismatch",
            "full_minus_label_only": paired_delta(rows, "type_mismatch", "symbolic", "label_only", args.seed),
            "full_gain": summaries["symbolic"]["axis"]["type_mismatch"]["source_minus_V"],
            "type_only_gain": summaries["type_only"]["axis"]["type_mismatch"]["source_minus_V"],
            "label_only_gain": summaries["label_only"]["axis"]["type_mismatch"]["source_minus_V"],
        },
    }
    symbolic = summaries["symbolic"]
    p1 = symbolic["source_minus_V"][2] >= 0.0
    p2 = all(symbolic["axis"][bucket]["support_ok"] and symbolic["axis"][bucket]["source_minus_V"][1] > 0.0 for bucket in GROUNDING_BUCKETS)
    p4 = symbolic["jaccard"][0] <= args.unconditional_jaccard + args.jaccard_slack and symbolic["jaccard"][2] < args.kill_ortho_jaccard
    oracle_up = symbolic["oracle_ceiling"] > args.unconditional_oracle + 1e-9
    support_ok = all(symbolic["axis"][bucket]["support_ok"] for bucket in GROUNDING_BUCKETS)
    if not support_ok:
        verdict = "PENDING_SUPPORT"
        consequent = "200-slice grounding buckets have n<30; inspect numbers before expanding slice"
    elif not p1:
        verdict = "DAMAGE-REMAINS"
        consequent = "symbolic annotations are still significantly below V"
    elif not p2:
        verdict = "NO-GROUNDING-REPAIR"
        consequent = "symbolic annotations did not significantly repair both grounding buckets"
    elif not p4:
        verdict = "ORTHO-LOST"
        consequent = "symbolic annotations pushed Jaccard toward non-orthogonal behavior"
    elif not oracle_up:
        verdict = "NO-ORACLE-UP"
        consequent = "source is viable but did not raise oracle ceiling beyond unconditional V+A pool"
    else:
        verdict = "SYMBOLIC-VIABLE"
        consequent = "proceed to verifier selection after review"
    return {
        "n": len(rows),
        "source_summaries": summaries,
        "dissociation": dissociation,
        "predictions": {
            "P1_no_damage": p1,
            "P2_grounding_repair": p2,
            "P4_orthogonality": p4,
            "oracle_up": oracle_up,
            "support_ok": support_ok,
        },
        "verdict": verdict,
        "consequent": consequent,
        "a_skipped": verdict != "SYMBOLIC-VIABLE",
    }


def fmt_ci(values: Sequence[float]) -> str:
    return f"{values[0]:+.4f} [{values[1]:+.4f}, {values[2]:+.4f}]"


def render_summary(summary: Dict[str, Any], args: argparse.Namespace) -> str:
    s = summary["source_summaries"]["symbolic"]
    lines = [
        "# V+A_symbolic: A11y as Symbolic Complement to Vision",
        "",
        "## Gate Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        summary["consequent"],
        "",
        "## Setup",
        "",
        f"- split: `GUI-360 balanced test`",
        f"- slice states: `{summary['n']}`",
        f"- V baseline rows: `{args.v_rows}`",
        f"- real UIA field: `step.control_infos.uia_controls_info`",
        f"- prompt guard: `control_rect` / coordinates excluded from symbolic text",
        f"- variants: `symbolic`, `type_only`, `label_only`",
        f"- unconditional V+A Jaccard baseline: `{args.unconditional_jaccard:.4f}`",
        f"- prompt-source Jaccard baseline: `{PROMPT_JACCARD:.2f}`",
        f"- unconditional oracle baseline: `{args.unconditional_oracle:.4f}`",
        "",
        "## P1 Overall No Damage",
        "",
        "| source | n | V correct | source correct | source - V | api errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in ["symbolic", "type_only", "label_only"]:
        row = summary["source_summaries"][source]
        lines.append(f"| {source} | {row['n']} | {row['V_correct_rate']:.4f} | {row['source_correct_rate']:.4f} | {fmt_ci(row['source_minus_V'])} | {row['api_errors']} |")
    lines += [
        "",
        "## P2 Grounding-Bucket Repair",
        "",
        "| source | bucket | n | support | V correct | source correct | source - V |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for source in ["symbolic", "type_only", "label_only"]:
        for bucket in GROUNDING_BUCKETS:
            row = summary["source_summaries"][source]["axis"][bucket]
            lines.append(f"| {source} | {bucket} | {row['n']} | {row['support_ok']} | {row['V_correct_rate']:.4f} | {row['source_correct_rate']:.4f} | {fmt_ci(row['source_minus_V'])} |")
    lines += [
        "",
        "## P3 Mechanism Dissociation",
        "",
        "| prediction | comparison | n | delta | interpretation |",
        "|---|---|---:|---:|---|",
    ]
    p3a = summary["dissociation"]["P3a_label_to_far_miss"]
    p3b = summary["dissociation"]["P3b_type_to_type_mismatch"]
    d = p3a["full_minus_type_only"]
    lines.append(f"| P3a label -> far-miss | symbolic - type_only on far_miss | {d[3]} | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] | label channel should add far-miss repair |")
    d = p3b["full_minus_label_only"]
    lines.append(f"| P3b type -> type-mismatch | symbolic - label_only on type_mismatch | {d[3]} | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] | type channel should add type-mismatch repair |")
    lines += [
        "",
        "## P4 Orthogonality + Oracle",
        "",
        "| source | Jaccard | Jaccard disagreement subset | agreement | disagreement n | oracle ceiling |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in ["symbolic", "type_only", "label_only"]:
        row = summary["source_summaries"][source]
        lines.append(f"| {source} | {row['jaccard'][0]:.4f} [{row['jaccard'][1]:.4f}, {row['jaccard'][2]:.4f}] | {row['jaccard_disagreement_subset'][0]:.4f} | {row['agreement'][0]:.4f} [{row['agreement'][1]:.4f}, {row['agreement'][2]:.4f}] | {row['disagreement_n']} | {row['oracle_ceiling']:.4f} |")
    lines += [
        "",
        "## Unique Coverage: symbolic",
        "",
        "| cell | count | share |",
        "|---|---:|---:|",
    ]
    for key, val in sorted(s["unique_coverage"].items()):
        lines.append(f"| {key} | {val} | {val / max(s['n'], 1):.4f} |")
    lines += [
        "",
        "## B Checks",
        "",
        f"- P1 no damage: `{summary['predictions']['P1_no_damage']}`",
        f"- P2 grounding repair: `{summary['predictions']['P2_grounding_repair']}`",
        f"- P4 orthogonality: `{summary['predictions']['P4_orthogonality']}`",
        f"- oracle up vs 70.5%: `{summary['predictions']['oracle_up']}`",
        f"- n>=30 support: `{summary['predictions']['support_ok']}`",
        "",
        "## A Verifier Selection",
        "",
    ]
    if summary["a_skipped"]:
        lines.append("**SKIPPED_B_NOT_SYMBOLIC_VIABLE**")
        lines.append("")
        lines.append("Verifier selection is skipped until B reaches `SYMBOLIC-VIABLE`.")
    else:
        lines.append("A verifier selection should be run next; this script stops for review before A.")
    lines += ["", "## One-Line Consequent", "", summary["consequent"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced_data_dir", default="datasets/gui360-balanced/data")
    parser.add_argument("--raw_repo", default="vyokky/GUI-360")
    parser.add_argument("--raw_local_dir", default="datasets/GUI-360-raw-jsonl")
    parser.add_argument("--v_rows", default="outputs/candidate_orthogonality/modality_jaccard/slice200/per_state.jsonl")
    parser.add_argument("--output_dir", default="outputs/candidate_orthogonality/va_symbolic")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max_controls", type=int, default=512)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--unconditional_jaccard", type=float, default=UNCONDITIONAL_JACCARD)
    parser.add_argument("--unconditional_oracle", type=float, default=UNCONDITIONAL_ORACLE)
    parser.add_argument("--jaccard_slack", type=float, default=0.12)
    parser.add_argument("--kill_ortho_jaccard", type=float, default=0.80)
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    v_rows = read_jsonl(Path(args.v_rows))
    v_rows_by_id = {row["state_id"]: row for row in v_rows}
    all_states = read_balanced_states(args.balanced_data_dir, 0)
    rng = random.Random(args.seed)
    rng.shuffle(all_states)
    states = all_states[: args.limit]
    coverage = attach_controls(states, args.raw_repo, args.raw_local_dir, args.log_every)
    for state in states:
        annotate_mechanisms(state)
    states = [state for state in states if state.get("controls") and state["state_id"] in v_rows_by_id]
    if len(states) != args.limit:
        print(f"Warning: using {len(states)} states after matching V rows and controls", flush=True)
    rows = evaluate_states(args, states, v_rows_by_id)
    summary = summarize(rows, args)
    summary["coverage"] = coverage
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "per_state.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "args": vars(args)}, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "summary.md").write_text(render_summary(summary, args))
    print(f"Wrote {out_dir / 'summary.md'}")
    print(f"Wrote {out_dir / 'per_state.jsonl'}")
    print(f"B_GATE: {summary['verdict']} - {summary['consequent']}")


if __name__ == "__main__":
    main()
