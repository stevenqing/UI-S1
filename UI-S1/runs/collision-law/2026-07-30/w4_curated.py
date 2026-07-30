import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
W4_REPO = RUN_DIR / "w4_repo"
W4_SOURCE_REVISION = "955062cc68e20666f71c4930e6025f989393a326"
DATA_DIR = RUN_DIR / "w4_data/metadata"
IMAGE_ROOT = RUN_DIR / "w4_data/image_root"
UPSTREAM = ROOT / "runs/androidcontrol-rft/2026-07-29"
MODEL_CONFIG = {
    "ui-agile-3b": (UPSTREAM / "models/UI-AGILE-3B", "KDEGroup/UI-AGILE-3B", "84c28b06a7bda29a741139d64e227d176c0fb1c0"),
    "ui-agile-7b": (UPSTREAM / "models/UI-AGILE-7B", "KDEGroup/UI-AGILE", "de01366937b3c921f49ae1abe3b2c4a39b40ce8d"),
    "ui-r1-e-3b": (UPSTREAM / "models/UI-R1-E-3B", "LZXzju/Qwen2.5-VL-3B-UI-R1-E", "91c3e5f213ab3f42931e6398174f470c8500167f"),
    "gui-r1-3b": (UPSTREAM / "models/GUI-R1/GUI-R1-3B", "ritzzai/GUI-R1:GUI-R1-3B", "e74baccc4cfa77074e2d53e99a8244ab9fc2ca10"),
    "gui-r1-7b": (UPSTREAM / "models/GUI-R1/GUI-R1-7B", "ritzzai/GUI-R1:GUI-R1-7B", "e74baccc4cfa77074e2d53e99a8244ab9fc2ca10"),
}
DATA_CONFIG = {
    "low": ("android_control_low_bbox.json", 8377),
    "high": ("android_control_high_task-improved.json", 8377),
}


def sha256_file(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    if not (W4_REPO / ".git").is_dir():
        raise FileNotFoundError("pinned W4 source clone is missing")
    head = __import__("subprocess").check_output(["git", "-C", str(W4_REPO), "rev-parse", "HEAD"], text=True).strip()
    if head != W4_SOURCE_REVISION:
        raise ValueError(f"W4 source revision mismatch: {head}")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_template():
    spec = importlib.util.spec_from_file_location("collision_w4_prompt", W4_REPO / "src/eval/prompt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.template


def task_and_history(row, setting):
    if setting == "high":
        return row.get("revised_task", row["instruction"]), row.get("revised_memory", row.get("history", ""))
    return row["instruction"], row.get("history", "")


def run_inference(args):
    verify_sources()
    model_dir, model_name, revision = MODEL_CONFIG[args.model]
    filename, expected_rows = DATA_CONFIG[args.setting]
    data_path = DATA_DIR / filename
    rows = json.loads(data_path.read_text())
    if len(rows) != expected_rows:
        raise ValueError(f"W4 {args.setting} row count mismatch")
    if any(not (IMAGE_ROOT / row["image"]).is_file() for row in rows):
        raise FileNotFoundError("W4 image coverage is incomplete")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    existing = read_jsonl(args.output) if args.resume else []
    completed = {row["index"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError("duplicate W4 resumed indices")

    processor = AutoProcessor.from_pretrained(model_dir.resolve(), trust_remote_code=True, use_fast=False)
    model = LLM(
        model=str(model_dir.resolve()), trust_remote_code=True,
        tensor_parallel_size=1, dtype="bfloat16", max_model_len=8192,
        gpu_memory_utilization=0.65, kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        limit_mm_per_prompt={"image": 1}, enforce_eager=True,
    )
    sampling = SamplingParams(
        temperature=0.0, top_p=0.001, repetition_penalty=1.05,
        max_tokens=512, skip_special_tokens=False,
    )
    template = prompt_template()
    pending = [index for index in indices if index not in completed]
    with args.output.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.batch_size):
            batch_indices = pending[start:start + args.batch_size]
            requests, metadata = [], []
            for index in batch_indices:
                row = rows[index]
                image_path = IMAGE_ROOT / row["image"]
                image = Image.open(image_path).convert("RGB")
                task, history = task_and_history(row, args.setting)
                prompt = template.format(Question=task, past_actions=history)
                message = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
                model_prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                image_inputs, _, video_kwargs = process_vision_info(message, return_video_kwargs=True)
                requests.append({
                    "prompt": model_prompt, "multi_modal_data": {"image": image_inputs},
                    "mm_processor_kwargs": video_kwargs,
                })
                metadata.append({
                    "image_size": list(image.size),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "model_prompt_sha256": hashlib.sha256(model_prompt.encode()).hexdigest(),
                })
            responses = model.generate(requests, sampling_params=sampling, use_tqdm=False)
            for index, item_metadata, response in zip(batch_indices, metadata, responses):
                row = rows[index]
                result = {
                    "index": index, "setting": args.setting, "model": args.model,
                    "image": row["image"], "image_size": item_metadata["image_size"],
                    "instruction": row["instruction"], "history": row.get("history", ""),
                    "revised_task": row.get("revised_task"), "revised_memory": row.get("revised_memory"),
                    "gt_action": row["gt_action"], "gt_max_bbox": row.get("gt_max_bbox"),
                    "candidate_actions": row.get("candidate_actions", []),
                    "response": response.outputs[0].text,
                    "prompt_sha256": item_metadata["prompt_sha256"],
                    "model_prompt_sha256": item_metadata["model_prompt_sha256"],
                    "model_name": model_name, "model_revision": revision,
                    "source_revision": W4_SOURCE_REVISION,
                    "dataset_sha256": sha256_file(data_path),
                    "generation": "temperature_0_top_p_0.001_repetition_1.05_max_512",
                    "num_shards": args.num_shards, "shard_index": args.shard_index,
                    "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                }
                output.write(json.dumps(result, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


def load_official_utils():
    source = W4_REPO / "src/eval"
    sys.path.insert(0, str(source.resolve()))
    import utils
    return utils


def primary_action(row):
    return [row["gt_action"], row.get("gt_max_bbox")]


def all_actions(row):
    actions = [primary_action(row)]
    for candidate in row.get("candidate_actions", []):
        actions.append([candidate.get("action_type", ""), candidate.get("action_bounds", [])])
    return actions


def score_predictions(args):
    verify_sources()
    utils = load_official_utils()
    rows = read_jsonl(args.predictions)
    expected = DATA_CONFIG[args.setting][1]
    if args.require_complete and (len(rows) != expected or [row["index"] for row in rows] != list(range(expected))):
        raise ValueError(f"complete W4 scoring requires ordered 0..{expected - 1}")
    scores = []
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for row in rows:
            width, height = row["image_size"]
            if args.setting == "high":
                bbox, action_type, step = utils.calculate_multi_android(
                    all_actions(row), row["response"], width, height, use_distance=False,
                )
            else:
                bbox, action_type, step = utils.calculate_single_android(
                    primary_action(row), row["response"], width, height, use_distance=False,
                )
            scores.append({"bbox": bbox, "action_type": action_type, "step": step})
    grounding_values = [score["bbox"] for score in scores if score["bbox"] is not None]
    result = {
        "status": "PASS", "coverage": "COMPLETE" if len(rows) == expected else "PARTIAL",
        "rows": len(rows), "model": rows[0]["model"] if rows else None,
        "setting": args.setting, "source_revision": W4_SOURCE_REVISION,
        "type_accuracy": sum(score["action_type"] for score in scores) / len(scores),
        "grounding_accuracy": sum(grounding_values) / len(grounding_values),
        "grounding_rows": len(grounding_values),
        "step_sr": sum(score["step"] for score in scores) / len(scores),
        "step_successes": sum(score["step"] for score in scores),
        "predictions_sha256": sha256_file(args.predictions),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--model", choices=MODEL_CONFIG, required=True)
    infer.add_argument("--setting", choices=DATA_CONFIG, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--num-shards", type=int, default=4)
    infer.add_argument("--shard-index", type=int, required=True)
    infer.add_argument("--batch-size", type=int, default=16)
    infer.add_argument("--kv-cache-memory-bytes", type=int, default=2 * 1024**3)
    infer.add_argument("--limit", type=int)
    infer.add_argument("--resume", action="store_true")
    score = subparsers.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--setting", choices=DATA_CONFIG, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.command == "infer":
        run_inference(args)
    else:
        score_predictions(args)


if __name__ == "__main__":
    main()