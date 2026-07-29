import argparse
import hashlib
import json
import os
import sys
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "repo/eval/android_control"
sys.path.insert(0, str(EVAL_DIR))

import android_control
from evaluate_android_control import evaluate_android_control_action
from qwen_vl_utils import smart_resize


android_control.Qwen2VL = partial(
    android_control.Qwen2VL,
    gpu_memory_utilization=0.65,
)
android_control.smart_resize = partial(smart_resize, factor=28)
AndroidControl = android_control.AndroidControl


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_tool_call(output: str) -> dict:
    prediction = output.split("</think>")[-1]
    if "<tool_call>" in prediction:
        prediction = prediction.split("<tool_call>", 1)[1]
    else:
        prediction = (
            '{"name": "mobile_use", "arguments":'
            + prediction.split('{"name": "mobile_use", "arguments":', 1)[1]
        )
    if "</tool_call>" in prediction:
        prediction = prediction.split("</tool_call>", 1)[0]
    else:
        prediction = prediction.split("<conclusion>", 1)[0]
        prediction = prediction.rsplit("}}", 1)[0] + "}}"
    return json.loads(prediction.strip())["arguments"]


def format_prompt(messages: list[dict]) -> str:
    prompt = ""
    if messages[0]["role"] != "system":
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role in {"user", "human"}:
            content = content.replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
            )
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role in {"assistant", "gpt"}:
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return prompt + "<|im_start|>assistant\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("high", "low"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {predictions_path}")
    existing = read_jsonl(predictions_path)
    completed_ids = [row["question_id"] for row in existing]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("duplicate question IDs in resume file")
    completed = set(completed_ids)

    evaluator = AndroidControl(
        model_path=str(ROOT / "model/InfiGUI-R1-3B"),
        eval_file=str(ROOT / "data/extracted/android_control_test.json"),
        image_root=str(ROOT / "data/extracted"),
        output_dir=str(args.output_dir),
        eval_type=args.setting,
        thinking=True,
        tensor_parallel_size=2,
        max_num_seqs=2,
        seed=42,
        num_processes=1,
    )
    jobs = evaluator.generate_jobs()
    if len(jobs) != 8444 or [job["question_id"] for job in jobs] != list(range(8444)):
        raise ValueError("official job coverage/order mismatch")
    pending = [job for job in jobs if job["question_id"] not in completed]

    with predictions_path.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.chunk_size):
            batch = pending[start : start + args.chunk_size]
            prompt_hashes = {
                job["question_id"]: hashlib.sha256(
                    format_prompt(job["messages"]).encode()
                ).hexdigest()
                for job in batch
            }
            results = evaluator.inference(
                batch, temperature=0.0, max_tokens=4096, seed=42
            )
            if len(results) != len(batch):
                raise ValueError("image preprocessing dropped a job")
            for job in results:
                parse_error = None
                prediction = None
                type_match = False
                exact_match = False
                try:
                    prediction = parse_tool_call(job["llm_output"])
                    type_match, exact_match = evaluate_android_control_action(
                        prediction,
                        job["check_pams"],
                        job["width"],
                        job["height"],
                        job["resized_width"],
                        job["resized_height"],
                        pred_type="abs_resized",
                        gt_type="abs_resized",
                    )
                except Exception as error:
                    parse_error = f"{type(error).__name__}: {error}"
                row = {
                    "question_id": job["question_id"],
                    "episode_id": job["episode_id"],
                    "step_id": job["step_id"],
                    "image_path": job["image_path"],
                    "width": job["width"],
                    "height": job["height"],
                    "resized_width": job["resized_width"],
                    "resized_height": job["resized_height"],
                    "check_pams": job["check_pams"],
                    "prompt_sha256": prompt_hashes[job["question_id"]],
                    "response": job["llm_output"],
                    "prediction": prediction,
                    "parse_error": parse_error,
                    "type_match": bool(type_match),
                    "exact_match": bool(exact_match),
                    "setting": args.setting,
                    "model_revision": "7b0e1de35afb807c6bfa70a2b85df24cf298e73d",
                    "source_revision": "a4fca17809a4395ba1fe08d481bb82c790ea7236",
                    "dataset_revision": "92bf0d54e371474bff2a94dd93f087ec6940b54d",
                    "backend": "vllm-0.11.0",
                    "tensor_parallel_size": 2,
                    "gpu_memory_utilization": 0.65,
                    "smart_resize_factor": 28,
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "seed": 42,
                }
                output.write(json.dumps(row, ensure_ascii=True) + "\n")
                output.flush()
                os.fsync(output.fileno())

    rows = read_jsonl(predictions_path)
    summary = {
        "status": "COMPLETE" if len(rows) == 8444 else "PARTIAL",
        "setting": args.setting,
        "rows": len(rows),
        "unique_question_ids": len({row["question_id"] for row in rows}),
        "parse_success": sum(row["parse_error"] is None for row in rows),
        "type_matches": sum(row["type_match"] for row in rows),
        "exact_matches": sum(row["exact_match"] for row in rows),
    }
    (args.output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()