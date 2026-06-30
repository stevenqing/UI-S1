"""Runtime helpers for format-matched capstone probes on ShareGPT arm data."""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from gui360_long_horizon.analysis.guards import assert_format_match
from gui360_long_horizon.data.longdep_pairs import LongDepPair, ShuffleControlError, read_pair_file
from gui360_long_horizon.experiments import v1_ood_repair, v2_conditionC_matched, v4_plan_recovery
from gui360_long_horizon.experiments import v3_longdep_pairs
from gui360_long_horizon.harness.correctness import function_match
from gui360_long_horizon.harness.model import VLLMClient
from gui360_long_horizon.harness.rollout import parse_tool_action, sharegpt_to_openai_messages


ProbeName = Literal["matched", "v1", "v2", "v3", "v4"]


@dataclass(frozen=True)
class TurnRecord:
    example_id: int
    turn_index: int
    step_uid: str
    current_human: Dict[str, str]
    target_assistant: Dict[str, str]
    matched_conversations: List[Dict[str, str]]
    matched_images: List[str]
    none_conversations: List[Dict[str, str]]
    none_images: List[str]

    @property
    def target_action(self) -> Dict[str, Any]:
        return parse_tool_action(str(self.target_assistant.get("value") or ""))


def _assert_sharegpt_example(example: Dict[str, Any], example_id: int) -> None:
    conversations = example.get("conversations") or []
    images = example.get("images") or []
    if len(conversations) % 2 != 0:
        raise ValueError(f"example {example_id} has odd number of conversation turns")
    markers = sum(str(turn.get("value", "")).count("<image>") for turn in conversations if turn.get("from") == "human")
    if markers != len(images):
        raise ValueError(f"example {example_id} image-count mismatch: markers={markers}, images={len(images)}")
    for index in range(0, len(conversations), 2):
        if conversations[index].get("from") != "human" or conversations[index + 1].get("from") != "gpt":
            raise ValueError(f"example {example_id} must alternate human/gpt turns")


def load_turn_records(dataset_json: str | Path, *, limit: int = -1) -> List[TurnRecord]:
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"ShareGPT dataset must be a list: {dataset_json}")
    records: List[TurnRecord] = []
    for example_id, example in enumerate(data):
        _assert_sharegpt_example(example, example_id)
        conversations = list(example.get("conversations") or [])
        images = list(example.get("images") or [])
        image_cursor = 0
        for turn_index in range(0, len(conversations), 2):
            human = conversations[turn_index]
            assistant = conversations[turn_index + 1]
            marker_count = str(human.get("value") or "").count("<image>")
            if marker_count <= 0:
                raise ValueError(f"example {example_id} turn {turn_index // 2} has no image marker")
            current_images = images[image_cursor:image_cursor + marker_count]
            matched_images = images[:image_cursor + marker_count]
            matched_conversations = conversations[:turn_index] + [human]
            records.append(
                TurnRecord(
                    example_id=example_id,
                    turn_index=turn_index // 2,
                    step_uid=f"ex{example_id}:turn{turn_index // 2}",
                    current_human=human,
                    target_assistant=assistant,
                    matched_conversations=matched_conversations,
                    matched_images=matched_images,
                    none_conversations=[human],
                    none_images=current_images,
                )
            )
            image_cursor += marker_count
            if limit > 0 and len(records) >= limit:
                return records
    return records


def _target_xy(action: Dict[str, Any]) -> Optional[tuple[float, float]]:
    coord = action.get("coordinate") or action.get("xy")
    if coord is None:
        raw = action.get("raw_json") if isinstance(action.get("raw_json"), dict) else {}
        args = raw.get("args") if isinstance(raw.get("args"), dict) else raw
        coord = args.get("coordinate") or args.get("xy")
    if coord is None:
        return None
    try:
        return float(coord[0]), float(coord[1])
    except (TypeError, ValueError, IndexError):
        return None


def _action_text(action: Dict[str, Any]) -> str:
    raw = action.get("raw_json") if isinstance(action.get("raw_json"), dict) else {}
    args = action.get("args") if isinstance(action.get("args"), dict) else raw.get("args") if isinstance(raw.get("args"), dict) else {}
    for source in (args, raw, action):
        for key in ("text", "keys", "value", "query", "control_text"):
            if source.get(key) is not None:
                return str(source.get(key))
    return ""


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def action_match(pred_action: Dict[str, Any], target_action: Dict[str, Any], *, coord_tol: float = 20.0) -> bool:
    target_function = str(target_action.get("function") or target_action.get("action") or "")
    if not function_match(pred_action, target_function):
        return False
    if str(target_function).strip().lower() in {"type", "input", "paste"}:
        target_text = _action_text(target_action)
        if target_text and _norm_text(_action_text(pred_action)) != _norm_text(target_text):
            return False
    target_xy = _target_xy(target_action)
    if target_xy is None:
        return True
    pred_xy = _target_xy(pred_action)
    if pred_xy is None:
        return False
    return math.dist(pred_xy, target_xy) <= coord_tol


def _decode(client: VLLMClient, conversations: List[Dict[str, str]], images: List[str], *, max_tokens: int, temperature: float) -> str:
    messages = sharegpt_to_openai_messages(conversations, images)
    return client.generate(messages, n=1, max_tokens=max_tokens, temperature=temperature)[0].text


def _oracle_plan(records: Sequence[TurnRecord]) -> str:
    lines = []
    for record in records:
        text = str(record.current_human.get("value") or "").replace("<image>", "").strip().splitlines()
        summary = " ".join(line.strip() for line in text if line.strip())[:200]
        if summary:
            lines.append(f"Step {record.turn_index}: {summary}")
    return "\n".join(lines)


def _with_plan(record: TurnRecord, plan: str) -> TurnRecord:
    human = dict(record.current_human)
    human["value"] = str(human.get("value") or "") + f"\n\nOracle/global plan:\n{plan}"
    matched = record.matched_conversations[:-1] + [human]
    none = [human]
    return TurnRecord(record.example_id, record.turn_index, record.step_uid, human, record.target_assistant, matched, record.matched_images, none, record.none_images)


def _with_injected_error(record: TurnRecord) -> Optional[TurnRecord]:
    if len(record.matched_conversations) < 3:
        return None
    corrupted = [dict(turn) for turn in record.matched_conversations]
    for index, turn in enumerate(corrupted):
        if turn.get("from") == "gpt":
            turn["value"] = '<tool_call>{"function":"click","args":{"coordinate":[0,0]},"status":"CONTINUE"}</tool_call>'
            return TurnRecord(record.example_id, record.turn_index, record.step_uid, record.current_human, record.target_assistant, corrupted, record.matched_images, record.none_conversations, record.none_images)
    return None


def _replace_dependency(record: TurnRecord, old: str, new: str) -> TurnRecord:
    if not old or old == new:
        return record
    replaced = []
    pattern = re.compile(re.escape(old), flags=re.IGNORECASE)
    for turn in record.matched_conversations:
        item = dict(turn)
        item["value"] = pattern.sub(new, str(item.get("value") or ""))
        replaced.append(item)
    return replace(record, matched_conversations=replaced)


ScoreInput = Tuple[TurnRecord, List[Dict[str, str]], List[str]]


def _score_record(client: VLLMClient, record: TurnRecord, conversations: List[Dict[str, str]], images: List[str], *, max_tokens: int, temperature: float, coord_tol: float) -> tuple[str, bool]:
    text = _decode(client, conversations, images, max_tokens=max_tokens, temperature=temperature)
    pred_action = parse_tool_action(text)
    return text, action_match(pred_action, record.target_action, coord_tol=coord_tol)


def _score_inputs(
    client: VLLMClient,
    inputs: Sequence[ScoreInput],
    *,
    max_tokens: int,
    temperature: float,
    coord_tol: float,
    workers: int = 1,
) -> List[tuple[str, bool]]:
    if workers <= 1 or len(inputs) <= 1:
        return [
            _score_record(client, record, conversations, images, max_tokens=max_tokens, temperature=temperature, coord_tol=coord_tol)
            for record, conversations, images in inputs
        ]
    outputs: List[tuple[str, bool] | None] = [None] * len(inputs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_score_record, client, record, conversations, images, max_tokens=max_tokens, temperature=temperature, coord_tol=coord_tol): index
            for index, (record, conversations, images) in enumerate(inputs)
        }
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
    return [output for output in outputs if output is not None]


def _distance_acc(rows: Sequence[Dict[str, Any]], distance: str) -> tuple[float, int]:
    values = [bool(row.get("step_correct", row.get("correct"))) for row in rows if (row.get("cond") or {}).get("distance") == distance and row.get("ok", True)]
    return (sum(values) / len(values), len(values)) if values else (0.0, 0)


def score_v3_pairs(
    records: Sequence[TurnRecord],
    client: VLLMClient,
    *,
    arm: str,
    history_format: str,
    pairs: Sequence[LongDepPair],
    shuffle_pairs: Sequence[LongDepPair] = (),
    max_tokens: int = 256,
    temperature: float = 0.0,
    coord_tol: float = 20.0,
    workers: int = 1,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    assert_format_match(arm, history_format)
    lookup = {record.step_uid: record for record in records}
    pending: List[tuple[LongDepPair, str, TurnRecord, bool]] = []

    for pair in pairs:
        near = lookup.get(pair.near_step_uid)
        far = lookup.get(pair.far_step_uid)
        if near is None or far is None:
            raise KeyError(f"pair {pair.pair_id} references missing step(s): {pair.near_step_uid}, {pair.far_step_uid}")
        pending.append((pair, "near", near, False))
        pending.append((pair, "far", far, False))

    for pair in shuffle_pairs:
        near = lookup.get(pair.near_step_uid)
        far = lookup.get(pair.far_step_uid)
        if near is None or far is None:
            raise KeyError(f"shuffle pair {pair.pair_id} references missing step(s): {pair.near_step_uid}, {pair.far_step_uid}")
        original = str((pair.metadata or {}).get("original_dependency_key") or "")
        replacement = str((pair.metadata or {}).get("shuffled_dependency_key") or pair.dependency_key)
        near = _replace_dependency(near, original, replacement)
        far = _replace_dependency(far, original, replacement)
        pending.append((pair, "near", near, True))
        pending.append((pair, "far", far, True))

    scored = _score_inputs(
        client,
        [(record, record.matched_conversations, record.matched_images) for _, _, record, _ in pending],
        max_tokens=max_tokens,
        temperature=temperature,
        coord_tol=coord_tol,
        workers=workers,
    )
    rows: List[Dict[str, Any]] = []
    for (pair, distance, record, shuffled), (text, correct) in zip(pending, scored):
        rows.append({
            "step_uid": record.step_uid,
            "example_id": record.example_id,
            "turn_index": record.turn_index,
            "arm": arm,
            "probe": "v3",
            "condition": f"{distance}{'_shuffle' if shuffled else ''}",
            "ok": True,
            "correct": correct,
            "step_correct": correct,
            "cond": {"history_format": history_format, "distance": distance, "pair_id": pair.pair_id, "dependency_key": pair.dependency_key, "shuffled": shuffled},
            "pred_text": text,
            "target_text": str(record.target_assistant.get("value") or ""),
        })

    shuffle_rows = [row for row in rows if row["cond"].get("shuffled")]
    shuffle_gap = 0.0
    if shuffle_rows:
        near_vals = [bool(row["step_correct"]) for row in shuffle_rows if row["cond"]["distance"] == "near"]
        far_vals = [bool(row["step_correct"]) for row in shuffle_rows if row["cond"]["distance"] == "far"]
        near_acc = sum(near_vals) / len(near_vals) if near_vals else 0.0
        far_acc = sum(far_vals) / len(far_vals) if far_vals else 0.0
        shuffle_gap = near_acc - far_acc
    real_rows = [row for row in rows if not row["cond"].get("shuffled")]
    try:
        summary = v3_longdep_pairs.summarize(real_rows, arm=arm, history_format=history_format, pairs=pairs, shuffle_gap=shuffle_gap).__dict__
    except ShuffleControlError as exc:
        near_acc, n_near = _distance_acc(real_rows, "near")
        far_acc, n_far = _distance_acc(real_rows, "far")
        summary = {
            "arm": arm,
            "history_format": history_format,
            "near_acc": near_acc,
            "far_acc": far_acc,
            "near_minus_far": near_acc - far_acc,
            "shuffle_gap": shuffle_gap,
            "shuffle_clean": False,
            "n_near": n_near,
            "n_far": n_far,
            "warning": str(exc),
        }
    return rows, summary


def score_probe(
    records: Sequence[TurnRecord],
    client: VLLMClient,
    *,
    arm: str,
    history_format: str,
    probe: ProbeName,
    max_tokens: int = 256,
    temperature: float = 0.0,
    coord_tol: float = 20.0,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    assert_format_match(arm, history_format)
    by_example: Dict[int, List[TurnRecord]] = {}
    for record in records:
        by_example.setdefault(record.example_id, []).append(record)

    targets: List[tuple[str, TurnRecord, List[Dict[str, str]], List[str], Dict[str, Any]]] = []
    for record in records:
        if probe == "v1":
            targets.append(("none", record, record.none_conversations, record.none_images, {"history_format": "none"}))
            targets.append((history_format, record, record.matched_conversations, record.matched_images, {"history_format": history_format}))
        elif probe == "matched":
            targets.append((history_format, record, record.matched_conversations, record.matched_images, {"history_format": history_format}))
        elif probe == "v2":
            targets.append(("clean", record, record.matched_conversations, record.matched_images, {"history_format": history_format, "injected_error": 0}))
            injected = _with_injected_error(record)
            if injected is not None:
                targets.append(("injected", injected, injected.matched_conversations, injected.matched_images, {"history_format": history_format, "injected_error": 1}))
        elif probe == "v4":
            plan = _oracle_plan(by_example.get(record.example_id, [record]))
            planned = _with_plan(record, plan)
            targets.append(("none", record, record.matched_conversations, record.matched_images, {"history_format": history_format, "plan": "none"}))
            targets.append(("oracle", planned, planned.matched_conversations, planned.matched_images, {"history_format": history_format, "plan": "oracle"}))
        else:
            raise ValueError(f"unknown probe: {probe}")

    scored = _score_inputs(
        client,
        [(target_record, conversations, images) for _, target_record, conversations, images, _ in targets],
        max_tokens=max_tokens,
        temperature=temperature,
        coord_tol=coord_tol,
        workers=workers,
    )
    rows: List[Dict[str, Any]] = []
    for (condition, target_record, _, _, cond), (text, correct) in zip(targets, scored):
        rows.append({
            "step_uid": target_record.step_uid,
            "example_id": target_record.example_id,
            "turn_index": target_record.turn_index,
            "arm": arm,
            "probe": probe,
            "condition": condition,
            "ok": True,
            "correct": correct,
            "step_correct": correct,
            "cond": cond,
            "pred_text": text,
            "target_text": str(target_record.target_assistant.get("value") or ""),
        })
    return rows


def summarize_probe(rows: Iterable[Dict[str, Any]], *, probe: ProbeName, arm: str, history_format: str) -> Dict[str, Any]:
    if probe == "matched":
        data = [row for row in rows if row.get("ok", True)]
        correct = [bool(row.get("step_correct", row.get("correct"))) for row in data]
        by_example: Dict[int, List[Dict[str, Any]]] = {}
        for row in data:
            by_example.setdefault(int(row["example_id"]), []).append(row)
        successes = 0
        progress_sum = 0.0
        for example_rows in by_example.values():
            ordered = sorted(example_rows, key=lambda item: int(item["turn_index"]))
            first_error = None
            for index, row in enumerate(ordered):
                if not bool(row.get("step_correct", row.get("correct"))):
                    first_error = index
                    break
            if first_error is None:
                successes += 1
                progress_sum += 1.0
            else:
                progress_sum += first_error / len(ordered) if ordered else 0.0
        n_examples = len(by_example)
        return {
            "arm": arm,
            "history_format": history_format,
            "step_acc": sum(correct) / len(correct) if correct else 0.0,
            "acc_matched": sum(correct) / len(correct) if correct else 0.0,
            "tsr": successes / n_examples if n_examples else 0.0,
            "avg_progress": progress_sum / n_examples if n_examples else 0.0,
            "n_steps": len(correct),
            "n_episodes": n_examples,
        }
    if probe == "v1":
        return v1_ood_repair.summarize(rows, arm=arm, history_format=history_format).__dict__
    if probe == "v2":
        return v2_conditionC_matched.summarize(rows, arm=arm, history_format=history_format).__dict__
    if probe == "v4":
        return v4_plan_recovery.summarize(rows, arm=arm, history_format=history_format).__dict__
    raise ValueError(f"unknown probe: {probe}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run format-matched capstone matched/V1/V2/V3/V4 probes on ShareGPT arm data")
    parser.add_argument("--dataset", required=True, help="ShareGPT validation JSON, e.g. gui360_gt_history_val.json")
    parser.add_argument("--probe", choices=["matched", "v1", "v2", "v3", "v4"], required=True)
    parser.add_argument("--arm", choices=["G", "O", "gt_history", "own_history"], required=True)
    parser.add_argument("--history-format", choices=["gt_history", "own_history"], required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-rows", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--pairs", default="", help="V3 long-dependency pair file")
    parser.add_argument("--shuffle-pairs", default="", help="Optional V3 shuffle-control pair file")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--coord-tol", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    records = load_turn_records(args.dataset, limit=args.limit)
    client = VLLMClient(args.base_url, args.model)
    if args.probe == "v3":
        if not args.pairs:
            raise SystemExit("--pairs is required for V3")
        pairs = read_pair_file(args.pairs)
        shuffle_pairs = read_pair_file(args.shuffle_pairs) if args.shuffle_pairs else []
        rows, summary = score_v3_pairs(records, client, arm=args.arm, history_format=args.history_format, pairs=pairs, shuffle_pairs=shuffle_pairs, max_tokens=args.max_tokens, temperature=args.temperature, coord_tol=args.coord_tol, workers=args.workers)
    else:
        rows = score_probe(records, client, arm=args.arm, history_format=args.history_format, probe=args.probe, max_tokens=args.max_tokens, temperature=args.temperature, coord_tol=args.coord_tol, workers=args.workers)
        summary = summarize_probe(rows, probe=args.probe, arm=args.arm, history_format=args.history_format)
    rows_path = Path(args.out_rows)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = Path(args.out_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": str(rows_path), "summary": str(summary_path), "n_rows": len(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()