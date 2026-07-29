import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor

from infer import GENERATION_CONFIG, build_messages, prompt_prefix, read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-family", choices=("qwen2", "qwen2.5"), required=True)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--setting", choices=("high", "low"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = read_jsonl(args.data)
    predictions = read_jsonl(args.predictions)
    score = json.loads(args.score.read_text())
    if len(predictions) != len(expected):
        raise ValueError(f"coverage mismatch: predictions={len(predictions)} expected={len(expected)}")

    identities = [row["identity"] for row in predictions]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate prediction identities")
    if score["status"] != "PASS":
        raise ValueError("score status is not PASS")
    for key in ("upstream_exact_parser", "flexible_parser_diagnostic"):
        if score[key]["rows"] != len(expected):
            raise ValueError(f"score coverage mismatch for {key}")

    processor = AutoProcessor.from_pretrained(args.model_dir.resolve(), use_fast=False)
    prefix = prompt_prefix()
    expected_generation = GENERATION_CONFIG
    if args.backend == "vllm":
        expected_generation = {
            "max_tokens": 128,
            "temperature": 0.01,
            "top_k": 1,
            "top_p": 0.001,
            "seed": 0,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
    for index, (source, prediction) in enumerate(zip(expected, predictions)):
        for key in ("identity", "episode_id", "step_id", "gt_action"):
            if prediction[key] != source[key]:
                raise ValueError(f"{key} mismatch at row {index}")
        if prediction["model_name"] != args.model_name:
            raise ValueError(f"model name mismatch at row {index}")
        if prediction["model_revision"] != args.model_revision:
            raise ValueError(f"model revision mismatch at row {index}")
        if prediction["model_family"] != args.model_family:
            raise ValueError(f"model family mismatch at row {index}")
        if prediction["processor_use_fast"] is not False:
            raise ValueError(f"processor implementation mismatch at row {index}")
        if prediction["generation"] != expected_generation:
            raise ValueError(f"generation configuration mismatch at row {index}")
        if args.backend == "vllm" and prediction.get("backend") != "vllm-0.11.0":
            raise ValueError(f"backend mismatch at row {index}")
        if prediction["setting"] != args.setting:
            raise ValueError(f"setting mismatch at row {index}")
        messages = build_messages(source, prefix, args.setting)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_hash = hashlib.sha256(text.encode()).hexdigest()
        if prediction["prompt_sha256"] != prompt_hash:
            raise ValueError(f"prompt hash mismatch at row {index}")

    summary = {
        "status": "PASS",
        "rows": len(predictions),
        "unique_identities": len(set(identities)),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "model_family": args.model_family,
        "backend": args.backend,
        "setting": args.setting,
        "data_sha256": sha256(args.data),
        "predictions_sha256": sha256(args.predictions),
        "score_sha256": sha256(args.score),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()