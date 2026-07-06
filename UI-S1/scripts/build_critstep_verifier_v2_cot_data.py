#!/usr/bin/env python3
"""Build Stage-1 GenRM-CoT verifier SFT data from the full-action verifier data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


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


def strip_old_tail(prompt: str) -> str:
    marker = "Return a brief reason, then a final line exactly one of:"
    if marker in prompt:
        return prompt.split(marker, 1)[0].rstrip()
    return prompt.rstrip()


def stage1_prompt_from_pointwise(prompt: str) -> str:
    base = strip_old_tail(prompt)
    return (
        f"{base}\n\n"
        "Return short verification reasoning under exactly these three lines:\n"
        "Type: does the candidate action type match the intended next step?\n"
        "Target: does the target UIA control text/type/geometry match the instruction referent?\n"
        "Content: if the action types text or presses a key, is the content right?\n"
        "Then finish with a final line exactly one of:\n"
        "VERDICT: Yes\n"
        "VERDICT: No"
    )


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


def parse_control_metadata(prompt: str) -> Dict[str, str]:
    block = extract_block(prompt, "Candidate UIA control metadata:\n", "\n\nReturn")
    out: Dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    if block.startswith("No UIA control"):
        out["no_control"] = block
    return out


def normalize_label(example: Mapping[str, Any]) -> str:
    metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
    label = str(metadata.get("label") or "").strip().lower()
    if label in {"correct", "yes", "true", "1"}:
        return "yes"
    if label in {"incorrect", "no", "false", "0"}:
        return "no"
    conversations = example.get("conversations") if isinstance(example.get("conversations"), list) else []
    assistant = conversations[1].get("value", "") if len(conversations) > 1 and isinstance(conversations[1], dict) else ""
    return "yes" if "VERDICT: correct" in assistant else "no"


def content_summary(action: Mapping[str, Any]) -> str:
    for key in ("text", "content", "value", "input", "key", "keys"):
        value = action.get(key)
        if value not in (None, "", []):
            return f"candidate content is {value!r}"
    return "no typed or key content is supplied"


def control_summary(control: Mapping[str, str]) -> str:
    if control.get("no_control"):
        return "no UIA control is assigned"
    control_type = control.get("control_type") or "unknown type"
    control_text = control.get("control_text") or "empty text"
    rect = control.get("control_rect") or "unknown rect"
    return f"UIA control type={control_type}, text={control_text!r}, rect={rect}"


def synthetic_rationale(prompt: str, label: str) -> str:
    action = parse_candidate_action(prompt)
    control = parse_control_metadata(prompt)
    action_type = str(action.get("action") or "unknown")
    target = control_summary(control)
    content = content_summary(action)
    if label == "yes":
        return (
            f"Type: The candidate action type is {action_type!r}, and it is compatible with the intended next GUI step.\n"
            f"Target: The candidate targets {target}; this target is consistent with the instruction referent and current screen.\n"
            f"Content: The candidate has {content}; this content is appropriate for the next full action.\n"
            "VERDICT: Yes"
        )
    return (
        f"Type: The candidate action type is {action_type!r}, but the full action is not the correct next GUI step.\n"
        f"Target: The candidate targets {target}; this target does not match the required target closely enough.\n"
        f"Content: The candidate has {content}; the action type, target, or content is insufficient for the instruction.\n"
        "VERDICT: No"
    )


def convert_example(example: Mapping[str, Any]) -> Dict[str, Any]:
    conversations = example.get("conversations") if isinstance(example.get("conversations"), list) else []
    if len(conversations) < 2:
        raise ValueError("expected sharegpt example with human and assistant turns")
    prompt = str(conversations[0].get("value") or "")
    label = normalize_label(example)
    converted = dict(example)
    converted["conversations"] = [
        {"from": "human", "value": stage1_prompt_from_pointwise(prompt)},
        {"from": "gpt", "value": synthetic_rationale(prompt, label)},
    ]
    metadata = dict(converted.get("metadata") or {})
    metadata["stage1_label"] = "Yes" if label == "yes" else "No"
    metadata["stage1_target_format"] = "genrm_cot_verdict_yes_no"
    converted["metadata"] = metadata
    return converted


def convert_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    rows = read_json(input_path)
    converted = [convert_example(row) for row in rows]
    write_json(output_path, converted)
    yes_count = sum(1 for row in converted if row.get("metadata", {}).get("stage1_label") == "Yes")
    return {"input": str(input_path), "output": str(output_path), "rows": len(converted), "yes": yes_count, "no": len(converted) - yes_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = convert_file(input_dir / "train_sft.json", output_dir / "train_genrm_cot_sft.json")
    val = convert_file(input_dir / "val_sft.json", output_dir / "val_genrm_cot_sft.json")
    dataset_info = {
        "critstep_genrm_cot_train": dataset_entry("train_genrm_cot_sft.json"),
        "critstep_genrm_cot_val": dataset_entry("val_genrm_cot_sft.json"),
    }
    write_json(output_dir / "dataset_info.json", dataset_info)
    write_json(output_dir / "dataset_info.snippet.json", dataset_info)
    manifest = {
        "source": str(input_dir),
        "train": train,
        "val": val,
        "no_leakage_note": "Inputs contain only instruction, screenshot, history, candidate action, and candidate UIA metadata. Labels and rationale targets are assistant outputs only.",
    }
    write_json(output_dir / "stage1_data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()