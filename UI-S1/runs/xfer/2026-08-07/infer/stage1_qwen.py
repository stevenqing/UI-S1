import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from xfer_common import MIND2WEB_ACTIONS, parse_product_response, parse_uitars_response


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate stage1 ids")
    return set(ids)


def prompt_text(roster, row):
    contract = roster["mind2web"]["prompt_contract"]
    history = "\n".join(
        str(item.get("step_repr") or item.get("operation") or item)
        for item in row["step_history"][-contract["history_steps"]:]
    ) or "None"
    return contract["user_template"].format(task=row["task"], history=history)


def load_model(model_type, model_dir):
    if model_type == "tongui":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="cuda:0",
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_dir, min_pixels=256 * 28 * 28, max_pixels=1344 * 28 * 28,
            model_max_length=8196, use_fast=False,
        )
        return model, processor
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_dir, size={"shortest_edge": 3136, "longest_edge": 2116800},
        min_pixels=3136, max_pixels=2116800, use_fast=False,
    )
    return model, processor


def infer(image, prompt, processor, model, model_type):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
    response = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    try:
        prediction = (
            parse_product_response(response, MIND2WEB_ACTIONS)
            if model_type == "tongui"
            else parse_uitars_response(response, MIND2WEB_ACTIONS)
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        prediction = {"action": None, "value": None, "position": None, "parse_ok": False}
    return response, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=("tongui", "uitars"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    model_spec = next(model for model in roster["mind2web"]["models"] if model["id"] == args.model_id)
    if Path(model_spec["local_path"]).resolve() != args.model_dir.resolve():
        raise ValueError("model path differs from frozen roster")
    rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    model, processor = load_model(args.model_type, args.model_dir)
    index_hash = sha256_file(args.model_dir / "model.safetensors.index.json")
    with args.output.open("a", buffering=1) as output:
        for index in indices:
            row = rows[index]
            if row["id"] in completed:
                continue
            image = Image.open(ROOT / row["image"]).convert("RGB")
            response, prediction = infer(image, prompt_text(roster, row), processor, model, args.model_type)
            artifact = {
                "stable_index": index,
                "id": row["id"],
                "annot_id": row["annot_id"],
                "action_uid": row["action_uid"],
                "image_sha256": row["image_sha256"],
                "model_id": args.model_id,
                "model_revision": model_spec["revision"],
                "model_index_sha256": index_hash,
                "response": response,
                "prediction": prediction,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    print(json.dumps({"status": "PASS", "model": args.model_id, "completed": len(completed_ids(args.output))}))


if __name__ == "__main__":
    main()
