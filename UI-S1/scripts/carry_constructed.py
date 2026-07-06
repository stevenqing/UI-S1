#!/usr/bin/env python3
"""Constructed prefix contrast for carry vs distribution shift on GUI-360.

Conditions per target step:
- C0: standard GT prefix.
- C1: high-divergence zero-error paraphrased GT prefix.
- C2_k: inject k actual model wrong actions into an otherwise GT prefix.
- C3_k: no-error GT prefix with text perturbation chosen to match C2_k divergence.

The screen remains the GT screenshot at the target step. This isolates the text
history channel under teacher forcing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v13_gui_360.eval_gui360_template import _format_action_for_history, build_step_prompt, parse_tool_call
from v13_gui_360.reward import compute_step_reward


DEFAULT_CARRY_PER_STEP = "outputs/carry_test/per_step.jsonl"
DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_PRED_RESULTS = "outputs/gui360_history_ab/original_sft_template_pred_history_full_20260701/eval_results_20260701_085620.json"
DEFAULT_OUTPUT_DIR = "outputs/carry_constructed"
DEFAULT_MODEL = "gui360-fullparam-sft-step250"

write_lock = Lock()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_coord(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if value[0] is None or value[1] is None:
        return None
    try:
        return int(float(value[0])), int(float(value[1]))
    except (TypeError, ValueError):
        return None


def action_text(action: Optional[Mapping[str, Any]], step_id: int, style: str) -> str:
    if not isinstance(action, Mapping):
        return f"Step {step_id}: no parsed action"
    action_type = str(action.get("action") or "unknown")
    coord = valid_coord(action.get("coordinate"))
    end = valid_coord(action.get("endCoordinate") or action.get("end_coordinate"))
    text = str(action.get("text") or action.get("keys") or "")
    if style == "standard":
        return _format_action_for_history(dict(action), step_id)
    if style == "compact":
        if action_type == "click" and coord:
            return f"{step_id}. CLICK at ({coord[0]}, {coord[1]})"
        if action_type == "type":
            return f"{step_id}. ENTER_TEXT value={json.dumps(text[:60], ensure_ascii=False)}"
        if action_type == "swipe" and coord and end:
            return f"{step_id}. SWIPE from ({coord[0]}, {coord[1]}) to ({end[0]}, {end[1]})"
        return f"{step_id}. {action_type.upper()}"
    if style == "verbose":
        if action_type == "click" and coord:
            return f"Previously, at step {step_id}, the cursor clicked the UI element located around x={coord[0]}, y={coord[1]}."
        if action_type == "type":
            return f"Previously, at step {step_id}, text was entered into the focused field: {json.dumps(text[:80], ensure_ascii=False)}."
        if action_type == "swipe" and coord and end:
            return f"Previously, at step {step_id}, the screen was dragged from x={coord[0]}, y={coord[1]} toward x={end[0]}, y={end[1]}."
        return f"Previously, at step {step_id}, the action type was {action_type}."
    if style == "jsonish":
        payload = {"step": step_id, "action": action_type}
        if coord:
            payload["coordinate"] = [coord[0], coord[1]]
        if end:
            payload["endCoordinate"] = [end[0], end[1]]
        if text:
            payload["text"] = text[:80]
        return "HistoryAction " + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    raise ValueError(style)


def build_history(actions: Sequence[Optional[Mapping[str, Any]]], styles: Optional[Mapping[int, str]] = None) -> List[str]:
    styles = styles or {}
    return [action_text(action, idx + 1, styles.get(idx, "standard")) for idx, action in enumerate(actions)]


def divergence(a: Sequence[str], b: Sequence[str]) -> float:
    left = "\n".join(a)
    right = "\n".join(b)
    if not left and not right:
        return 0.0
    return 1.0 - SequenceMatcher(None, left, right).ratio()


def choose_c3_history(gt_actions: Sequence[Mapping[str, Any]], source_steps: Sequence[int], target_divergence: float) -> Tuple[List[str], float, str]:
    base = build_history(gt_actions)
    candidates: List[Tuple[str, List[str]]] = []
    for style in ("compact", "verbose", "jsonish"):
        candidates.append((f"source_{style}", build_history(gt_actions, {idx: style for idx in source_steps})))
    for style in ("compact", "verbose", "jsonish"):
        candidates.append((f"all_{style}", build_history(gt_actions, {idx: style for idx in range(len(gt_actions))})))
    if source_steps:
        prefix_until = max(source_steps) + 1
        for style in ("compact", "verbose", "jsonish"):
            candidates.append((f"prefix_to_source_{style}", build_history(gt_actions, {idx: style for idx in range(prefix_until)})))
    best_name, best_history = min(candidates, key=lambda item: abs(divergence(base, item[1]) - target_divergence))
    return best_history, divergence(base, best_history), best_name


def parse_action_from_pred_step(step: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    action = step.get("pred_action")
    return action if isinstance(action, dict) else None


def build_targets(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    carry_rows = read_jsonl(Path(args.carry_per_step))
    episodes = {str(row["episode_id"]): row for row in read_jsonl(Path(args.test_data))}
    pred_results = {str(value.get("episode_id", key)): value for key, value in load_json(Path(args.pred_results)).items()}
    by_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in carry_rows:
        by_ep[str(row["episode_id"])].append(row)
    eligible = []
    skipped = Counter()
    for episode_id, rows in by_ep.items():
        rows.sort(key=lambda item: int(item["step_idx"]))
        episode = episodes.get(episode_id)
        pred_episode = pred_results.get(episode_id)
        if not episode or not pred_episode:
            skipped["missing_episode"] += 1
            continue
        pred_steps = pred_episode.get("steps") if isinstance(pred_episode.get("steps"), list) else []
        gt_steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
        for row in rows:
            target_idx = int(row["step_idx"])
            if target_idx < 3 or target_idx >= len(gt_steps) or target_idx >= len(pred_steps):
                skipped["target_too_early_or_oob"] += 1
                continue
            if not bool(row.get("gt_correct")):
                skipped["target_gt_history_not_correct"] += 1
                continue
            wrong_prior = []
            for prior_idx in range(target_idx):
                if prior_idx >= len(pred_steps):
                    continue
                pred_step = pred_steps[prior_idx]
                pred_action = parse_action_from_pred_step(pred_step)
                if pred_action and not bool(pred_step.get("success")):
                    wrong_prior.append(prior_idx)
            if len(wrong_prior) < 3:
                skipped["not_enough_prior_wrong_actions"] += 1
                continue
            eligible.append((episode_id, target_idx, wrong_prior[-3:]))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(eligible)
    selected = eligible[: args.target_n]
    targets: List[Dict[str, Any]] = []
    for target_serial, (episode_id, target_idx, source_steps_last3) in enumerate(selected):
        episode = episodes[episode_id]
        pred_episode = pred_results[episode_id]
        gt_steps = episode["steps"]
        pred_steps = pred_episode["steps"]
        gt_prefix_actions = [gt_steps[idx]["action"] for idx in range(target_idx)]
        c0_history = build_history(gt_prefix_actions)
        c1_history = build_history(gt_prefix_actions, {idx: "jsonish" if idx % 2 else "verbose" for idx in range(target_idx)})
        base_payload = {
            "episode_id": episode_id,
            "target_step": target_idx,
            "target_uid": f"{episode_id}:{target_idx}",
            "target_serial": target_serial,
            "instruction": episode.get("goal", ""),
            "screenshot": gt_steps[target_idx].get("screenshot"),
            "gt_action": gt_steps[target_idx].get("action") if isinstance(gt_steps[target_idx].get("action"), dict) else {},
            "image_w": int(gt_steps[target_idx].get("image_w") or 1040),
            "image_h": int(gt_steps[target_idx].get("image_h") or 736),
            "target_gt_existing_correct": True,
        }
        def add(condition: str, k: int, history: List[str], source_steps: Sequence[int], error_count: int, c3_style: Optional[str] = None) -> None:
            targets.append({
                **base_payload,
                "condition": condition,
                "k": k,
                "condition_id": f"{episode_id}:{target_idx}:{condition}:k{k}",
                "history": history,
                "source_steps": list(source_steps),
                "error_count": error_count,
                "divergence_vs_c0": divergence(c0_history, history),
                "c3_style": c3_style,
            })
        add("C0_gt_prefix", 0, c0_history, [], 0)
        add("C1_highdiv_zero_error", 0, c1_history, [], 0, "all_mixed_verbose_jsonish")
        for k in (1, 2, 3):
            source_steps = source_steps_last3[-k:]
            injected_actions = list(gt_prefix_actions)
            for source_idx in source_steps:
                injected_actions[source_idx] = pred_steps[source_idx]["pred_action"]
            c2_history = build_history(injected_actions)
            c2_div = divergence(c0_history, c2_history)
            c3_history, _c3_div, c3_style = choose_c3_history(gt_prefix_actions, source_steps, c2_div)
            add("C2_lowdiv_error", k, c2_history, source_steps, k)
            add("C3_matched_no_error", k, c3_history, source_steps, 0, c3_style)
    manifest = {
        "target_n_requested": args.target_n,
        "eligible_targets": len(eligible),
        "selected_targets": len(selected),
        "condition_rows": len(targets),
        "conditions": dict(Counter(row["condition"] for row in targets)),
        "skipped": dict(skipped),
        "construction_note": "C2 injects actual prior pred-history wrong actions from the same episode; C3 chooses a no-error paraphrase closest to C2 text divergence.",
    }
    return targets, manifest


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("condition_id")) for row in read_jsonl(path)}


def score_prediction(pred_text: str, target: Mapping[str, Any]) -> Dict[str, Any]:
    pred_action = parse_tool_call(pred_text)
    fake_text = f"<action>{json.dumps(pred_action, ensure_ascii=False)}</action>" if pred_action else pred_text
    reward, info = compute_step_reward(fake_text, target["gt_action"], int(target["image_w"]), int(target["image_h"]))
    return {
        "success": bool(reward >= 0.5),
        "reward": float(reward),
        "pred_action": info.get("pred_action") or pred_action,
        "pred_type": info.get("pred_type"),
        "gt_type": info.get("gt_type"),
        "format_reward": info.get("format_reward", 0.0),
        "type_reward": info.get("type_reward", 0.0),
        "content_reward": info.get("content_reward", 0.0),
        "pred_text": pred_text[:700],
    }


def eval_one(target: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    messages = build_step_prompt(
        str(target["instruction"]),
        str(target["screenshot"]),
        int(target["target_step"]),
        list(target["history"]),
        image_max_pixels=args.image_max_pixels,
    )
    errors = []
    pred_text = ""
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            pred_text = response.choices[0].message.content or ""
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc)[:300])
            if attempt < args.max_retries:
                time.sleep(min(10.0, 1.5 ** attempt))
    scored = score_prediction(pred_text, target)
    return {k: v for k, v in target.items() if k != "history"} | scored | {"api_errors": errors, "api_error_count": len(errors)}


def run_eval(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "targets.jsonl"
    result_path = output_dir / "per_step.jsonl"
    targets, manifest = build_targets(args)
    write_json(output_dir / "manifest.json", manifest)
    if not target_path.exists() or args.force_targets:
        with target_path.open("w", encoding="utf-8") as handle:
            for row in targets:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    completed = done_ids(result_path) if args.resume else set()
    work = [target for target in targets if str(target["condition_id"]) not in completed]
    print(json.dumps({"targets": len(targets), "completed": len(completed), "remaining": len(work), "manifest": manifest}, indent=2), flush=True)
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(eval_one, target, args): target for target in work}
        for index, future in enumerate(as_completed(futures), 1):
            target = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {k: v for k, v in target.items() if k != "history"} | {"success": False, "reward": 0.0, "eval_error": str(exc)[:500]}
            append_jsonl(result_path, row)
            if index % 50 == 0 or index == len(work):
                print(f"completed {index}/{len(work)}", flush=True)
    summarize(args)


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def condition_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), int(row.get("k") or 0))].append(row)
    out = {}
    for (condition, k), group in sorted(grouped.items()):
        out[f"{condition}:k{k}"] = {
            "condition": condition,
            "k": k,
            "n": len(group),
            "success_rate": sum(1 for row in group if row.get("success")) / len(group) if group else None,
            "mean_reward": mean([float(row.get("reward") or 0.0) for row in group]),
            "mean_divergence_vs_c0": mean([float(row.get("divergence_vs_c0") or 0.0) for row in group]),
        }
    return out


def paired_drop(rows: Sequence[Mapping[str, Any]], condition_a: str, condition_b: str, k: int = 0) -> Dict[str, Any]:
    by_target = {(str(row["target_uid"]), str(row["condition"]), int(row.get("k") or 0)): row for row in rows}
    target_ids = sorted({str(row["target_uid"]) for row in rows})
    pairs = []
    for target_uid in target_ids:
        a = by_target.get((target_uid, condition_a, k))
        b = by_target.get((target_uid, condition_b, k))
        if a and b:
            pairs.append((a, b))
    if not pairs:
        return {"n": 0, "a_success": None, "b_success": None, "drop_a_minus_b": None}
    a_success = sum(1 for a, _ in pairs if a.get("success")) / len(pairs)
    b_success = sum(1 for _, b in pairs if b.get("success")) / len(pairs)
    return {
        "n": len(pairs),
        "a": condition_a,
        "b": condition_b,
        "k": k,
        "a_success": a_success,
        "b_success": b_success,
        "drop_a_minus_b": a_success - b_success,
        "a_gt_only": sum(1 for a, b in pairs if a.get("success") and not b.get("success")),
        "b_better": sum(1 for a, b in pairs if b.get("success") and not a.get("success")),
        "mean_divergence_a": mean([float(a.get("divergence_vs_c0") or 0.0) for a, _ in pairs]),
        "mean_divergence_b": mean([float(b.get("divergence_vs_c0") or 0.0) for _, b in pairs]),
        "mean_abs_divergence_gap": mean([abs(float(a.get("divergence_vs_c0") or 0.0) - float(b.get("divergence_vs_c0") or 0.0)) for a, b in pairs]),
    }


def decide_gate(summary: Mapping[str, Any]) -> Dict[str, str]:
    shift = summary["shift_test"]["drop_a_minus_b"]
    dose = summary["carry_tests"]
    carry3 = dose.get("k3", {}).get("drop_a_minus_b")
    carry1 = dose.get("k1", {}).get("drop_a_minus_b")
    if carry3 is not None and carry3 >= 0.08 and (carry1 is None or carry3 >= carry1) and (shift is None or carry3 > shift + 0.03):
        return {"verdict": "CARRY", "reason": "Error-injected prefixes underperform divergence-matched no-error prefixes, with the largest effect at k=3."}
    if carry3 is not None and abs(carry3) <= 0.03 and shift is not None and shift >= 0.05:
        return {"verdict": "DISTRIBUTION SHIFT", "reason": "High-divergence no-error prefixes hurt, while matched error content adds little."}
    return {"verdict": "MIXED", "reason": "Both text-form and error-content effects are small or shared; report the decomposition."}


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Constructed Carry vs Distribution Shift Contrast", ""]
    lines.append("Teacher-forced GT screens. Conditions construct history text only; frozen matcher; no training.")
    lines.append("")
    lines.append("## Condition Counts")
    lines.append("")
    lines.append("| condition | k | n | success | mean reward | mean divergence |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in summary["condition_summary"].values():
        lines.append(f"| {item['condition']} | {item['k']} | {item['n']} | {pct(item['success_rate'])} | {item['mean_reward']:.3f} | {item['mean_divergence_vs_c0']:.3f} |")
    lines.append("")
    lines.append("## SHIFT Test: C0 vs C1")
    lines.append("")
    shift = summary["shift_test"]
    lines.append("| comparison | n | C0 success | C1 success | C0-C1 drop | C0-only | C1-better |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| high-div zero-error | {shift['n']} | {pct(shift['a_success'])} | {pct(shift['b_success'])} | {pct(shift['drop_a_minus_b'])} | {shift.get('a_gt_only', 0)} | {shift.get('b_better', 0)} |")
    lines.append("")
    lines.append("## CARRY Test: C2(k-error) vs C3(matched no-error)")
    lines.append("")
    lines.append("| k | n | C3 success | C2 success | C3-C2 error-content effect | mean C2 div | mean C3 div | mean abs div gap |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ["k1", "k2", "k3"]:
        item = summary["carry_tests"].get(key, {})
        lines.append(f"| {item.get('k')} | {item.get('n', 0)} | {pct(item.get('a_success'))} | {pct(item.get('b_success'))} | {pct(item.get('drop_a_minus_b'))} | {item.get('mean_divergence_a', 0.0):.3f} | {item.get('mean_divergence_b', 0.0):.3f} | {item.get('mean_abs_divergence_gap', 0.0):.3f} |")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']['verdict']}**")
    lines.append("")
    lines.append(summary["gate"]["reason"])
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'constructed.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append(f"- `{output_dir / 'targets.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows = read_jsonl(output_dir / "per_step.jsonl")
    if not rows:
        raise SystemExit("no per_step rows to summarize")
    summary: Dict[str, Any] = {
        "condition_summary": condition_summary(rows),
        "shift_test": paired_drop(rows, "C0_gt_prefix", "C1_highdiv_zero_error", 0),
        "carry_tests": {
            "k1": paired_drop(rows, "C3_matched_no_error", "C2_lowdiv_error", 1),
            "k2": paired_drop(rows, "C3_matched_no_error", "C2_lowdiv_error", 2),
            "k3": paired_drop(rows, "C3_matched_no_error", "C2_lowdiv_error", 3),
        },
        "n_rows": len(rows),
        "n_targets": len({row["target_uid"] for row in rows}),
    }
    summary["gate"] = decide_gate(summary)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "constructed.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "n_rows": len(rows), "n_targets": summary["n_targets"], "gate": summary["gate"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carry-per-step", default=DEFAULT_CARRY_PER_STEP)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--pred-results", default=DEFAULT_PRED_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--api-url", default="http://127.0.0.1:8142/v1")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-targets", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()