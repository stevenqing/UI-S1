import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from pka import Prediction
from selfconsistency import self_consistency_product_space


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[3]
ASSET_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets"
MODEL_DIR = ASSET_ROOT / "GTA1-7B"
DATA_DIR = ASSET_ROOT / "ScreenSpot-Pro"
MODEL_REVISION = "701bedc80b447863bd60e3318ae44f6cbbfafd78"
DATA_REVISION = "210e78d3844251110bff86c95835ebd37a6930fa"
MODEL_CARD_ANCHOR = 0.501
SOURCE_README_UPDATED_ANCHOR = 0.555
SYSTEM_PROMPT = """You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)"""


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def samples():
    output = []
    for path in sorted((DATA_DIR / "annotations").glob("*.json")):
        for row in json.loads(path.read_text()):
            output.append({"annotation_file": path.name, **row})
    if len(output) != 1581 or len({row["id"] for row in output}) != 1581:
        raise ValueError("ScreenSpot-Pro annotation coverage mismatch")
    if any(not (DATA_DIR / "images" / row["img_filename"]).is_file() for row in output):
        raise FileNotFoundError("ScreenSpot-Pro image coverage incomplete")
    return output


def parse_coordinate(response: str):
    matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", response)
    if not matches:
        return None
    try:
        return tuple(map(float, matches[0]))
    except ValueError:
        return None


def load_completed(path):
    if not path.exists():
        return set()
    indices = [json.loads(line)["index"] for line in path.read_text().splitlines() if line.strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate GTA1 resumed indices")
    return set(indices)


def aggregate_points(points, width, height):
    predictions = [
        Prediction("CLICK", point[0] / width, point[1] / height, source=f"sample_{index}")
        for index, point in enumerate(points) if point is not None
    ]
    aggregate = self_consistency_product_space(predictions)
    if aggregate is None or aggregate.coordinate is None:
        return (0.0, 0.0), False
    return (aggregate.x * width, aggregate.y * height), True


def infer(args):
    rows = samples()
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, min_pixels=3136, max_pixels=4096 * 2160,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0",
    ).eval()
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = load_completed(args.output) if args.resume else set()
    with args.output.open("a", buffering=1) as output:
        for index in indices:
            if index in completed:
                continue
            row = rows[index]
            image_path = DATA_DIR / "images" / row["img_filename"]
            image = Image.open(image_path).convert("RGB")
            resized_height, resized_width = smart_resize(
                image.height, image.width,
                factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
                min_pixels=processor.image_processor.min_pixels,
                max_pixels=processor.image_processor.max_pixels,
            )
            resized = image.resize((resized_width, resized_height))
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)},
                {"role": "user", "content": [
                    {"type": "image", "image": resized},
                    {"type": "text", "text": row["instruction"]},
                ]},
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)
            responses = []
            parsed_points = []
            for sample_index in range(args.samples):
                seed = args.seed + index * args.samples + sample_index
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                generation_args = {
                    "max_new_tokens": 32,
                    "do_sample": args.temperature > 0,
                    "use_cache": True,
                }
                if args.temperature > 0:
                    generation_args["temperature"] = args.temperature
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, **generation_args)
                generated = output_ids[:, inputs.input_ids.shape[1]:]
                response = processor.batch_decode(
                    generated, skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
                responses.append(response)
                parsed_points.append(parse_coordinate(response))
            compatibility_point, parse_ok = aggregate_points(
                parsed_points, resized_width, resized_height,
            )
            original_point = (
                compatibility_point[0] * image.width / resized_width,
                compatibility_point[1] * image.height / resized_height,
            )
            result = {
                "index": index, "id": row["id"], "annotation_file": row["annotation_file"],
                "img_filename": row["img_filename"], "img_size": row["img_size"],
                "bbox": row["bbox"], "instruction": row["instruction"],
                "ui_type": row["ui_type"], "application": row["application"],
                "response": responses[0], "responses": responses, "parse_ok": parse_ok,
                "sample_parse_ok": [point is not None for point in parsed_points],
                "sample_points_resized": [list(point) if point is not None else None for point in parsed_points],
                "pred_point_resized": list(compatibility_point),
                "pred_point_original": list(original_point),
                "resized_size": [resized_width, resized_height],
                "model_revision": MODEL_REVISION, "dataset_revision": DATA_REVISION,
                "protocol": "pinned_model_card_locator_with_0_0_parse_fallback",
                "samples": args.samples, "temperature": args.temperature, "seed": args.seed,
                "num_shards": args.num_shards, "shard_index": args.shard_index,
            }
            output.write(json.dumps(result, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


def merge(args):
    rows = {}
    for shard in range(args.num_shards):
        path = args.shard_root / f"shard-{shard}.jsonl"
        for row in [json.loads(line) for line in path.read_text().splitlines() if line.strip()]:
            if row["index"] in rows or row["index"] % args.num_shards != shard:
                raise ValueError("GTA1 shard identity mismatch")
            rows[row["index"]] = row
    if set(rows) != set(range(1581)):
        raise ValueError("GTA1 merge requires indices 0..1580")
    with args.output.open("w") as output:
        for index in range(1581):
            output.write(json.dumps(rows[index], ensure_ascii=True) + "\n")


def score(args):
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or [row["index"] for row in rows] != list(range(1581)):
        raise ValueError("GTA1 scoring requires ordered 1,581 rows")
    compatibility_correct = []
    strict_correct = []
    for row in rows:
        x0, y0, x1, y1 = row["bbox"]
        x, y = row["pred_point_original"]
        hit = x0 <= x <= x1 and y0 <= y <= y1
        compatibility_correct.append(hit)
        strict_correct.append(hit and row["parse_ok"])
    compatibility = sum(compatibility_correct) / len(rows)
    strict = sum(strict_correct) / len(rows)
    result = {
        "status": "PASS", "rows": len(rows),
        "model_card_compatibility_accuracy": compatibility,
        "strict_parse_accuracy": strict,
        "parse_rate": sum(row["parse_ok"] for row in rows) / len(rows),
        "reported_anchor_conflict": {
            "pinned_model_card": MODEL_CARD_ANCHOR,
            "source_readme_updated": SOURCE_README_UPDATED_ANCHOR,
            "delta_to_model_card": compatibility - MODEL_CARD_ANCHOR,
            "delta_to_source_readme": compatibility - SOURCE_README_UPDATED_ANCHOR,
        },
        "predictions_sha256": sha256_file(args.predictions),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inference = sub.add_parser("infer")
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--num-shards", type=int, default=4)
    inference.add_argument("--shard-index", type=int, required=True)
    inference.add_argument("--limit", type=int)
    inference.add_argument("--resume", action="store_true")
    inference.add_argument("--samples", type=int, default=1)
    inference.add_argument("--temperature", type=float, default=0.0)
    inference.add_argument("--seed", type=int, default=20260730)
    merger = sub.add_parser("merge")
    merger.add_argument("--shard-root", type=Path, required=True)
    merger.add_argument("--num-shards", type=int, default=4)
    merger.add_argument("--output", type=Path, required=True)
    scorer = sub.add_parser("score")
    scorer.add_argument("--predictions", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"infer": infer, "merge": merge, "score": score}[args.command](args)


if __name__ == "__main__":
    main()