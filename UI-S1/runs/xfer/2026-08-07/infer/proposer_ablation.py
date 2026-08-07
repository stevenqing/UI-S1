import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
MVP_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"
sys.path.insert(0, str(MVP_ROOT))

from mvp_sspro import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    rank_regions_by_coverage,
)


IMAGE_START = 151652
IMAGE_END = 151653
COMMA = 11


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
        raise ValueError("duplicate proposer ablation ids")
    return set(ids)


def prompt_text(config, row):
    history = "\n".join(
        str(item.get("step_repr") or item.get("operation") or item)
        for item in row["step_history"][-config["mind2web"]["prompt_contract"]["history_steps"]:]
    ) or "None"
    return config["mind2web"]["prompt_contract"]["user_template"].format(
        task=row["task"], history=history
    )


def generate_multilayer(image, prompt, processor, model, layers, proposer):
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized = image.resize((resized_width, resized_height))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": resized},
        {"type": "text", "text": prompt},
    ]}]
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    captured = {}
    for layer in layers:
        def hook(query_states, key_states, layer=layer):
            if model.should_capture:
                captured[layer] = (query_states.detach().clone(), key_states.detach().clone())
        model.model.layers[layer].self_attn.register_attention_hook(hook)
    model.target_token_id = COMMA
    model.target_layer_idx = 0
    model.clear_captured_states()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
    for layer in layers:
        model.model.layers[layer].self_attn.register_attention_hook(None)
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    if set(captured) != set(layers):
        raise ValueError(f"missing layer captures; response={response!r}; captured={sorted(captured)}")
    input_ids = inputs["input_ids"][0]
    starts = torch.where(input_ids.eq(IMAGE_START))[0]
    ends = torch.where(input_ids.eq(IMAGE_END))[0]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("unexpected visual token boundaries")
    image_start = int(starts[0])
    image_end = int(ends[0])
    layer_regions = {}
    for layer, (query_states, key_states) in captured.items():
        attention = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(query_states.shape[-1])
        weights = torch.softmax(attention.squeeze(2), dim=-1).max(dim=1)[0][0, image_start + 1:image_end]
        top_k = min(proposer["attention"]["top_k"], len(weights))
        top_indices = torch.topk(weights, top_k).indices.cpu().numpy()
        patch_size = proposer["attention"]["patch_size"]
        patches_w = resized_width // patch_size
        positions = [
            (
                (int(index) % patches_w + 0.5) * patch_size * image.width / resized_width,
                (int(index) // patches_w + 0.5) * patch_size * image.height / resized_height,
            )
            for index in top_indices
        ]
        ranked = rank_regions_by_coverage(
            positions,
            image.width,
            image.height,
            proposer["attention"]["max_regions"],
            proposer["crop_geometry"]["width"],
            proposer["crop_geometry"]["height"],
        )
        layer_regions[str(layer)] = [
            {
                "region": [int(round(value)) for value in item["region"]],
                "coverage": int(item["coverage"]),
            }
            for item in ranked
        ]
    return response, [resized_width, resized_height], layer_regions


def full_bbox_hit(region, bbox):
    left, top, right, bottom = region
    return left <= bbox["x"] and top <= bbox["y"] and right >= bbox["x"] + bbox["width"] and bottom >= bbox["y"] + bbox["height"]


def center_hit(region, bbox):
    x = bbox["x"] + bbox["width"] / 2
    y = bbox["y"] + bbox["height"] / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def summarize(rows, layers):
    report = {}
    for layer in layers:
        full_by_rank = defaultdict(list)
        center_by_rank = defaultdict(list)
        for row in rows:
            for rank, value in enumerate(row["layers"][str(layer)]):
                full_by_rank[rank].append(full_bbox_hit(value["region"], row["target_bbox"]))
                center_by_rank[rank].append(center_hit(value["region"], row["target_bbox"]))
        full_rates = [sum(full_by_rank[rank]) / len(full_by_rank[rank]) for rank in range(12)]
        center_rates = [sum(center_by_rank[rank]) / len(center_by_rank[rank]) for rank in range(12)]
        report[str(layer)] = {
            "rank0_full_bbox_containment": full_rates[0],
            "mean_rank0_to_rank11_full_bbox_containment": float(np.mean(full_rates)),
            "full_bbox_containment_by_rank": full_rates,
            "center_containment_by_rank": center_rates,
        }
    selected = max(
        layers,
        key=lambda layer: (
            report[str(layer)]["rank0_full_bbox_containment"],
            report[str(layer)]["mean_rank0_to_rank11_full_bbox_containment"],
            -layer,
        ),
    )
    return report, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("mind2web",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    proposer = roster[args.benchmark]["proposer"]
    if proposer["selection_status"] != "PENDING_DEV_ABLATION":
        raise ValueError("proposer selection is not pending")
    layers = list(proposer["layer_candidates"])
    rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    folds = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    selected_rows = [row for row in rows if folds[row["website"]] == proposer["dev_fold"]]
    if len(selected_rows) != 416:
        raise ValueError(f"expected 416 Mind2Web dev rows, found {len(selected_rows)}")
    if args.limit is not None:
        selected_rows = selected_rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    model_spec = next(model for model in roster[args.benchmark]["models"] if model["id"] == proposer["model"])
    model_dir = ROOT / model_spec["local_path"]
    config = Qwen2_5_VLConfig.from_pretrained(model_dir)
    config.target_token_id = ","
    config.target_layer_idx = 0
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0",
    ).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_dir, min_pixels=256 * 28 * 28, max_pixels=1344 * 28 * 28,
    )
    with args.output.open("a", buffering=1) as output:
        for row in selected_rows:
            if row["id"] in completed:
                continue
            image = Image.open(ROOT / row["image"]).convert("RGB")
            response, resized_size, layer_regions = generate_multilayer(
                image, prompt_text(roster, row), processor, model, layers, proposer
            )
            artifact = {
                "id": row["id"],
                "stable_index": row["stable_index"],
                "website": row["website"],
                "image_sha256": row["image_sha256"],
                "target_bbox": row["step"]["bbox"],
                "response": response,
                "resized_size": resized_size,
                "layers": layer_regions,
            }
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    all_rows = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
    if args.limit is None and len(all_rows) == 416:
        report, selected = summarize(all_rows, layers)
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps({
            "schema_version": 1,
            "status": "PASS",
            "benchmark": args.benchmark,
            "dev_fold": proposer["dev_fold"],
            "rows": len(all_rows),
            "layers": report,
            "selected_layer": selected,
            "query_token": proposer["query_token_candidates"][0],
            "model_revision": model_spec["revision"],
            "model_index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
            "trace_sha256": sha256_file(args.output),
        }, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "PASS", "selected_layer": selected, "summary": str(summary_path)}, indent=2))
    else:
        print(json.dumps({"status": "PARTIAL", "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
