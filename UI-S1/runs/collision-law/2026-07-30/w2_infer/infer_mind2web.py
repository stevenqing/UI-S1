import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data._utils.collate import default_collate

from views import generate_view, max_visual_tokens


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[3]
UPSTREAM_DIR = ROOT / "runs/mind2web-tongui/2026-07-28"
MODEL_DIR = UPSTREAM_DIR / "models/TongUI-7B"
MODEL_NAME = "TongUI-7B"
MODEL_REVISION = "a3e0cf46c3164bbd885dea2694f2ad7a31f1661d"
FULL_PREDICTIONS = UPSTREAM_DIR / "artifacts/tongui-7b/merged/predictions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def load_completed(path: Path):
    if not path.exists():
        return set()
    values = [json.loads(line)["index"] for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError("duplicate resumed Mind2Web indices")
    return set(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("v1", "v2", "v3", "v4"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index outside range")

    repo_dir = UPSTREAM_DIR / "repos/TongUI-agent"
    sys.path.insert(0, str(repo_dir.resolve()))
    from tongui.data.dset_mind2web import Mind2WebDataset, get_answer
    from tongui.data.template import mind2web_to_qwen
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    visual_tokens = max_visual_tokens("mind2web", args.view)
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR.resolve(), min_pixels=256 * 28 * 28,
        max_pixels=visual_tokens * 28 * 28, model_max_length=8196, use_fast=False,
    )
    full_rows = read_jsonl(FULL_PREDICTIONS)
    if len(full_rows) != 2080 or [row["index"] for row in full_rows] != list(range(2080)):
        raise ValueError("TongUI full-view trace is incomplete")

    score_path = UPSTREAM_DIR / "score.py"
    import importlib.util
    score_spec = importlib.util.spec_from_file_location("collision_tongui_score", score_path)
    if score_spec is None or score_spec.loader is None:
        raise ImportError(score_path)
    score_module = importlib.util.module_from_spec(score_spec)
    score_spec.loader.exec_module(score_module)

    class ViewDataset(Mind2WebDataset):
        def get_sample(self, index):
            item = copy.deepcopy(self.json_data[index % len(self.json_data)])
            original_path = Path(self.IMG_DIR) / item["img_url"]
            original = Image.open(original_path).convert("RGB")
            full_prediction = None
            try:
                parsed = score_module.parse_prediction(full_rows[index]["response"])
                position = parsed.get("position")
                if position is not None and len(position) == 2 and all(math.isfinite(value) for value in position):
                    full_prediction = (position[0] * original.width, position[1] * original.height)
            except (IndexError, KeyError, TypeError, ValueError):
                full_prediction = None
            generated = generate_view(original, args.view, full_prediction)

            history_images = []
            self.append_history_image(item, self.num_history, history_images, url_only=True)
            image_list = history_images + [generated.image]
            action_history = self.get_history_qwen(
                image_list, item, self.num_history, self.interleaved_history
            )
            answer = get_answer(item, item["step"], item["step_repr"])
            source = mind2web_to_qwen(item["task"], action_history, None, version=self.version)
            prompt = self.processor.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(source)
            processed = self.processor(
                text=[prompt], images=image_inputs, videos=video_inputs,
                return_tensors="pt", training=False,
            )
            if "labels" not in processed:
                processed["labels"] = processed["input_ids"]
            data = {
                "input_ids": processed["input_ids"][0],
                "image_sizes": processed["image_grid_thw"],
                "pixel_values": processed["pixel_values"],
                "labels": processed["input_ids"][0],
            }
            for key in ("select_mask", "patch_pos", "patch_assign", "patch_assign_len"):
                if key in processed:
                    data[key] = processed[key]
            meta = {
                "annot_id": item["annot_id"], "action_uid": item["action_uid"],
                "split": item["split"], "image": item["img_url"], "answer": answer,
                "bbox": item["step"]["bbox"], "image_size": item["img_size"],
                "view_size": list(generated.geometry.view_size),
                "view_offset": [generated.geometry.offset_x, generated.geometry.offset_y],
                "center_fallback": generated.geometry.center_fallback,
            }
            return data, meta

    dataset = ViewDataset(
        str((UPSTREAM_DIR / "data/evaluation_data").resolve()),
        "Mind2Web", "hf_test_task_with_thoughts", processor, inference=True,
        args_dict={"num_history": 2, "interleaved_history": "vtvt", "version": "v2"},
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR.resolve(), device_map="cuda:0", torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    indices = [index for index in range(len(dataset)) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    if output_path.exists() and not args.resume:
        raise FileExistsError(output_path)
    completed = load_completed(output_path) if args.resume else set()
    with output_path.open("a", buffering=1) as output:
        for index in indices:
            if index in completed:
                continue
            data, meta = dataset[index]
            inputs = default_collate([data])
            input_hash = tensor_sha256(inputs["input_ids"])
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to("cuda:0")
            inputs["pixel_values"] = inputs["pixel_values"].bfloat16()
            forward = {
                "pixel_values": inputs["pixel_values"], "input_ids": inputs["input_ids"],
                "labels": inputs["labels"], "image_grid_thw": inputs["image_sizes"].squeeze(0),
            }
            for key in ("patch_assign", "patch_assign_len", "patch_pos", "select_mask"):
                if key in inputs:
                    forward[key] = inputs[key]
            with torch.inference_mode():
                generated = model.generate(
                    **forward, max_new_tokens=128,
                    eos_token_id=processor.tokenizer.eos_token_id, do_sample=False,
                )
            generated = generated[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True,
            )[0]
            result = {
                "index": index, **meta, "response": response,
                "input_ids_sha256": input_hash, "model": "tongui-7b",
                "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                "view_id": args.view, "pred_source": f"tongui-7b__{args.view}",
                "max_visual_tokens": visual_tokens, "generation": "greedy",
                "num_history": 2, "interleaved_history": "vtvt",
                "shard_index": args.shard_index, "num_shards": args.num_shards,
                "full_prediction_source_sha256": hashlib.sha256(
                    json.dumps(full_rows[index], sort_keys=True).encode()
                ).hexdigest(),
            }
            output.write(json.dumps(result, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()