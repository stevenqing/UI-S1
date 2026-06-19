#!/usr/bin/env python3
"""Generate Verifier Agent JSON decisions from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

from evaluate_verifier_agent import parse_decision_text  # noqa: E402


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> list[JsonDict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def load_model(base_model: Path, checkpoint: Path, dtype: torch.dtype, device: str) -> tuple[Any, Any]:
    if (checkpoint / ".dcp_checkpoint").exists():
        raise ValueError(f"{checkpoint} is a sharded DCP checkpoint, not an inference-ready HF/PEFT checkpoint")
    tokenizer_source = checkpoint if (checkpoint / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    is_adapter = (checkpoint / "adapter_config.json").exists()
    if is_adapter:
        if PeftModel is None:
            raise ValueError("peft is required to load adapter checkpoints")
        base = AutoModelForCausalLM.from_pretrained(str(base_model), torch_dtype=dtype, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(checkpoint))
    else:
        model_source = checkpoint if (checkpoint / "config.json").exists() else base_model
        model = AutoModelForCausalLM.from_pretrained(str(model_source), torch_dtype=dtype, trust_remote_code=True)
    model.eval()
    model.to(device)
    return tokenizer, model


def prompt_from_messages(messages: list[JsonDict]) -> list[JsonDict]:
    if not messages:
        return []
    if messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def generate_one(tokenizer: Any, model: Any, messages: list[JsonDict], device: str, max_new_tokens: int) -> str:
    prompt_messages = prompt_from_messages(messages)
    encoded = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_len = encoded["input_ids"].shape[-1]
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
    generated = output[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Verifier Agent route decisions")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    tokenizer, model = load_model(Path(args.base_model), Path(args.checkpoint), dtype_from_name(args.dtype), device)
    rows = iter_jsonl(Path(args.data))
    if args.limit > 0:
        rows = rows[: args.limit]
    outputs = []
    for index, row in enumerate(rows):
        text = generate_one(tokenizer, model, row.get("messages", []), device, args.max_new_tokens)
        outputs.append({
            "index": index,
            "assistant": text,
            "decision": parse_decision_text(text),
            "target": row.get("target"),
            "metadata": row.get("metadata"),
        })
        if (index + 1) % 25 == 0:
            print(f"generated {index + 1}/{len(rows)}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, outputs)
    print(f"wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
