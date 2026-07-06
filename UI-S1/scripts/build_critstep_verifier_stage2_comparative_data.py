#!/usr/bin/env python3
"""Build Stage-2 comparative verifier SFT data from candidate-level verifier data."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_INPUT_DIR = "outputs/critstep_verifier"
DEFAULT_OUTPUT_DIR = "outputs/critstep_verifier_v2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dataset_entry(file_name: str) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
    }


def extract_block(prompt: str, start: str, end: str) -> str:
    if start not in prompt:
        return ""
    after = prompt.split(start, 1)[1]
    if end in after:
        return after.split(end, 1)[0].strip()
    return after.strip()


def parse_candidate_action(prompt: str) -> Dict[str, Any]:
    text = extract_block(prompt, "Candidate action JSON:\n", "\n\nCandidate UIA control metadata:")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_control_metadata_text(prompt: str) -> str:
    return extract_block(prompt, "Candidate UIA control metadata:\n", "\n\nReturn")


def prompt_instruction(prompt: str) -> str:
    return extract_block(prompt, "Instruction:\n", "\n\nAction history:")


def prompt_history(prompt: str) -> str:
    return extract_block(prompt, "Action history:\n", "\n\nCandidate action JSON:") or "None"


def normalize_label(example: Mapping[str, Any]) -> str:
    metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
    label = str(metadata.get("label") or "").strip().lower()
    if label in {"correct", "yes", "true", "1"}:
        return "correct"
    if label in {"incorrect", "no", "false", "0"}:
        return "incorrect"
    conversations = example.get("conversations") if isinstance(example.get("conversations"), list) else []
    assistant = conversations[1].get("value", "") if len(conversations) > 1 and isinstance(conversations[1], dict) else ""
    return "correct" if "VERDICT: correct" in assistant else "incorrect"


def action_signature(action: Mapping[str, Any], control_text: str) -> str:
    payload = {"action": action, "control": control_text}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def parse_pointwise_example(example: Mapping[str, Any]) -> Dict[str, Any]:
    conversations = example.get("conversations") if isinstance(example.get("conversations"), list) else []
    if len(conversations) < 2:
        raise ValueError("expected sharegpt example with human and assistant turns")
    prompt = str(conversations[0].get("value") or "")
    metadata = dict(example.get("metadata") or {})
    action = parse_candidate_action(prompt)
    control_text = parse_control_metadata_text(prompt)
    label = normalize_label(example)
    return {
        "target_id": str(metadata.get("target_id") or ""),
        "episode_id": str(metadata.get("episode_id") or ""),
        "step_idx": metadata.get("step_idx"),
        "candidate_id": str(metadata.get("candidate_id") or ""),
        "candidate_source": str(metadata.get("candidate_source") or ""),
        "subset": str(metadata.get("subset") or ""),
        "depth_bin": str(metadata.get("depth_bin") or ""),
        "label": label,
        "instruction": prompt_instruction(prompt),
        "history": prompt_history(prompt),
        "action": action,
        "control_text": control_text,
        "image": (example.get("images") or [None])[0],
        "distinct_key": action_signature(action, control_text),
    }


def candidate_block(label: str, candidate: Mapping[str, Any]) -> str:
    action_text = json.dumps(candidate.get("action") or {}, ensure_ascii=False, sort_keys=True)
    return (
        f"Candidate {label} action JSON:\n{action_text}\n\n"
        f"Candidate {label} UIA control metadata:\n{candidate.get('control_text') or 'No UIA control metadata.'}"
    )


def comparative_prompt(instruction: str, history: str, cand_a: Mapping[str, Any], cand_b: Mapping[str, Any]) -> str:
    return (
        "<image>\n"
        "You are a comparative verifier for GUI actions. Given the current screenshot, user instruction, action history, "
        "and exactly two candidate next actions, choose which candidate is the better next FULL ACTION.\n\n"
        "Compare action type, target UIA control text/type/geometry, coordinates if relevant, and typed/key content if relevant. "
        "Use only the screenshot, instruction, history, and the two candidate actions with their UIA metadata. Do not assume candidate frequency, rank, source, or labels.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Action history:\n{history or 'None'}\n\n"
        f"{candidate_block('A', cand_a)}\n\n"
        f"{candidate_block('B', cand_b)}\n\n"
        "Return short comparison reasoning under these lines:\n"
        "Type: compare whether A or B has the action type required by the intended next step.\n"
        "Target: compare which target control/text/geometry better matches the instruction referent.\n"
        "Content: compare typed/key content if applicable.\n"
        "Then finish with a final line exactly one of:\n"
        "WINNER: A\n"
        "WINNER: B"
    )


def comparative_target(winner: str, cand_a: Mapping[str, Any], cand_b: Mapping[str, Any]) -> str:
    loser = "B" if winner == "A" else "A"
    return (
        f"Type: Candidate {winner} better matches the action type required by the intended next step; candidate {loser} is a distractor on type, target, or full-action fit.\n"
        f"Target: Candidate {winner}'s UIA target/action is the correct match for the instruction referent in this screen state, while candidate {loser}'s target/action is less appropriate.\n"
        f"Content: Candidate {winner} has the correct full-action content if content is required; candidate {loser} should not be selected for this next step.\n"
        f"WINNER: {winner}"
    )


def dedupe_candidates(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row["distinct_key"])
        current = by_key.get(key)
        if current is None:
            by_key[key] = dict(row)
            by_key[key]["candidate_ids"] = [row.get("candidate_id")]
            by_key[key]["sources"] = [row.get("candidate_source")]
            continue
        current["candidate_ids"].append(row.get("candidate_id"))
        current["sources"].append(row.get("candidate_source"))
        if row.get("label") == "correct":
            current["label"] = "correct"
    return list(by_key.values())


def select_pairs(candidates: Sequence[Mapping[str, Any]], other_negatives_per_positive: int) -> List[Tuple[Mapping[str, Any], Mapping[str, Any], str]]:
    positives = [cand for cand in candidates if cand.get("label") == "correct"]
    negatives = [cand for cand in candidates if cand.get("label") != "correct"]
    pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any], str]] = []
    seen = set()
    for pos in positives:
        greedy = [neg for neg in negatives if "greedy" in set(neg.get("sources") or [])]
        other = [neg for neg in negatives if neg not in greedy]
        selected = greedy[:1] + other[: max(0, other_negatives_per_positive)]
        for neg in selected:
            key = (pos["distinct_key"], neg["distinct_key"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append((pos, neg, "greedy" if neg in greedy else "other"))
    return pairs


def build_examples(input_path: Path, output_path: Path, *, other_negatives_per_positive: int, include_swapped: bool, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    raw = read_json(input_path)
    parsed = [parse_pointwise_example(row) for row in raw]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        grouped[str(row["target_id"])].append(row)
    examples: List[Dict[str, Any]] = []
    target_count = 0
    pair_count = 0
    greedy_pair_count = 0
    for target_id, rows in sorted(grouped.items()):
        distinct = dedupe_candidates(rows)
        pairs = select_pairs(distinct, other_negatives_per_positive)
        if not pairs:
            continue
        target_count += 1
        base = rows[0]
        for pair_idx, (pos, neg, pair_type) in enumerate(pairs):
            orientations = [(pos, neg, "A")]
            if include_swapped:
                orientations.append((neg, pos, "B"))
            else:
                if (hash((target_id, pair_idx, seed)) % 2) == 1:
                    orientations = [(neg, pos, "B")]
            for cand_a, cand_b, winner in orientations:
                prompt = comparative_prompt(str(base.get("instruction") or ""), str(base.get("history") or "None"), cand_a, cand_b)
                examples.append({
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": comparative_target(winner, cand_a, cand_b)},
                    ],
                    "images": [base.get("image")],
                    "metadata": {
                        "target_id": target_id,
                        "episode_id": base.get("episode_id"),
                        "step_idx": base.get("step_idx"),
                        "subset": base.get("subset"),
                        "depth_bin": base.get("depth_bin"),
                        "pair_type": pair_type,
                        "winner": winner,
                        "positive_candidate_ids": pos.get("candidate_ids"),
                        "negative_candidate_ids": neg.get("candidate_ids"),
                        "leakage_note": "source/labels/candidate ids are excluded from prompt text; metadata is for audit only.",
                    },
                })
            pair_count += 1
            greedy_pair_count += int(pair_type == "greedy")
    write_json(output_path, examples)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(examples),
        "targets": target_count,
        "base_pairs": pair_count,
        "greedy_base_pairs": greedy_pair_count,
        "include_swapped": include_swapped,
        "other_negatives_per_positive": other_negatives_per_positive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--other-negatives-per-positive", type=int, default=4)
    parser.add_argument("--include-swapped", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = build_examples(
        input_dir / "train_sft.json",
        output_dir / "train_stage2_comparative_sft.json",
        other_negatives_per_positive=args.other_negatives_per_positive,
        include_swapped=args.include_swapped,
        seed=args.seed,
    )
    val = build_examples(
        input_dir / "val_sft.json",
        output_dir / "val_stage2_comparative_sft.json",
        other_negatives_per_positive=args.other_negatives_per_positive,
        include_swapped=args.include_swapped,
        seed=args.seed + 1,
    )
    dataset_info_path = output_dir / "dataset_info.json"
    dataset_info = read_json(dataset_info_path) if dataset_info_path.exists() else {}
    dataset_info.update({
        "critstep_stage2_comparative_train": dataset_entry("train_stage2_comparative_sft.json"),
        "critstep_stage2_comparative_val": dataset_entry("val_stage2_comparative_sft.json"),
    })
    write_json(dataset_info_path, dataset_info)
    write_json(output_dir / "dataset_info.snippet.json", dataset_info)
    manifest = {
        "source": str(input_dir),
        "train": train,
        "val": val,
        "no_leakage_note": "Inputs contain only instruction, screenshot, history, and two candidate actions with UIA metadata. Pair labels are assistant targets only.",
    }
    write_json(output_dir / "stage2_comparative_data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()