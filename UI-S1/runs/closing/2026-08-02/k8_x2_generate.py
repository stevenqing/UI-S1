import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DIVERSITY_X2 = ROOT / "runs/diversity-axis/2026-08-02/x2"
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
sys.path.insert(0, str(DIVERSITY_X2))
from generate_microchains import (
    MODEL_SPECS,
    NORMALIZED_SYSTEM_PROMPT,
    PIXEL_SYSTEM_PROMPT,
    infer,
    load_backend,
    parse_pair,
    sha256_file,
)
from zoom_port import adaptive_crop, deterministic_seed, gate, point_to_box


def infer_many(image, instruction, model_type, processor, model, seed, count=8):
    resized_height, resized_width = smart_resize(
        image.height, image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=2000000, max_pixels=4800000,
    )
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    system_prompt = (
        PIXEL_SYSTEM_PROMPT.format(height=resized_height, width=resized_width)
        if model_type == "gta1" else NORMALIZED_SYSTEM_PROMPT
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": resized},
            {"type": "text", "text": instruction},
        ]},
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=32, do_sample=True, temperature=0.9,
            top_p=1.0, num_return_sequences=count, use_cache=True,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=pad_token_id,
        )
    generated_ids = generated.sequences[:, inputs.input_ids.shape[1]:]
    responses = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    transition = model.compute_transition_scores(generated.sequences, generated.scores, normalize_logits=True)
    output = []
    for index, response in enumerate(responses):
        token_ids = generated_ids[index, :transition.shape[1]]
        scores = transition[index]
        mask = token_ids.ne(pad_token_id) & torch.isfinite(scores)
        finite = scores[mask]
        confidence = float(torch.exp(finite.mean()).item()) if finite.numel() else 0.0
        parsed = parse_pair(response)
        if parsed is None:
            point = None
        elif model_type == "gta1":
            point = [parsed[0] * image.width / resized_width, parsed[1] * image.height / resized_height]
        elif 0 <= parsed[0] <= 1000 and 0 <= parsed[1] <= 1000:
            point = [parsed[0] / 1000 * image.width, parsed[1] / 1000 * image.height]
        else:
            point = None
        output.append({
            "sample_index": index,
            "point": point,
            "response": response,
            "confidence": confidence,
        })
    return output, [resized_width, resized_height]


def run_chain(source, image, family, model_type, model_id, chain_index, processor, model):
    width, height = image.size
    global_seed = deterministic_seed(source["id"], family, model_id, chain_index, "global_K8")
    globals_, resized_size = infer_many(
        image, source["instruction"], model_type, processor, model, global_seed, 8
    )
    candidates = []
    for value in globals_:
        value["seed"] = global_seed
        value["temperature"] = 0.9
        value["region"] = [0, 0, width, height]
        value["box"] = point_to_box(value["point"], width, height)
        candidates.append({"box": value["box"], "confidence": value["confidence"]})
    report = gate(candidates)
    crop = adaptive_crop(candidates, width, height) if not report["reliable"] else None
    confirmation_seed = deterministic_seed(source["id"], family, model_id, chain_index, "confirmation")
    confirmation = infer(
        image, source["instruction"], model_type, processor, model,
        confirmation_seed, 0.9,
    )
    confirmation.update({
        "seed": confirmation_seed,
        "region": [0, 0, width, height],
        "box": point_to_box(confirmation["point"], width, height),
    })
    refinement = None
    if crop is not None:
        left, top, right, bottom = crop
        refinement_seed = deterministic_seed(source["id"], family, model_id, chain_index, "refinement")
        refinement = infer(
            image.crop(crop), source["instruction"], model_type, processor, model,
            refinement_seed, 0.0, offset_x=left, offset_y=top, global_image=False,
        )
        refinement.update({
            "seed": refinement_seed,
            "region": crop,
            "box": point_to_box(refinement["point"], width, height),
        })
    report.update({
        "chain_index": chain_index,
        "crop": crop,
        "adaptive_uses": "refinement" if refinement is not None else "confirmation",
        "fixed_forwards": 9,
        "adaptive_forwards": 9,
        "union_generated_forwards": 10 if refinement is not None else 9,
    })
    return {
        "chain_index": chain_index,
        "global_seed": global_seed,
        "global_resized_size": resized_size,
        "global_K8": globals_,
        "confirmation": confirmation,
        "refinement": refinement,
        "report": report,
    }


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("K8 X2 duplicate resumed identities")
    return set(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--family", choices=("single", "mixed"), required=True)
    parser.add_argument("--model-type", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.family == "single" and args.model_type != "gta1":
        raise ValueError("K8 X2 single family uses GTA1 only")
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any("bbox" in row or "target_bbox" in row for row in rows):
        raise ValueError("K8 X2 requires complete label-free inputs")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    spec, model, processor = load_backend(args.model_type)
    model_hash = sha256_file(spec["path"] / "model.safetensors.index.json")
    chains = range(3) if args.family == "single" else range(1)
    written = 0
    with args.output.open("a", buffering=1) as output:
        for stable_index in indices:
            source = rows[stable_index]
            if source["id"] in completed:
                continue
            image = Image.open(DATA_ROOT / "images" / source["img_filename"]).convert("RGB")
            chain_values = [
                run_chain(source, image, args.family, args.model_type, spec["id"], chain, processor, model)
                for chain in chains
            ]
            artifact = {
                **source,
                "family": args.family,
                "model_id": spec["id"],
                "model_revision": spec["revision"],
                "model_index_sha256": model_hash,
                "policy": "K8_paired_fixed27",
                "chains": chain_values,
                "fixed_cell_forwards": 9 * len(chain_values),
                "adaptive_cell_forwards": 9 * len(chain_values),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            artifact["chains_sha256"] = hashlib.sha256(
                json.dumps(chain_values, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            written += 1
            if written % 25 == 0:
                print(json.dumps({"family": args.family, "model": spec["id"], "shard": args.shard_index, "written": written}), flush=True)
    print(json.dumps({"status": "PASS", "family": args.family, "model": spec["id"], "shard": args.shard_index, "written": written}), flush=True)


if __name__ == "__main__":
    main()