import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image
import yaml
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from g1_generate import canonical_hash, parse_response, prompt_for, sha256_file


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
ROSTER_PATH = RUN_DIR / "configs/g1_roster.yaml"


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("G2 duplicate resumed score identities")
    return set(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--model", choices=("GTA1-72B", "UI-Venus-Ground-72B", "Qwen3.5-122B-A10B"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.68)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    region_rows = [json.loads(line) for line in args.regions.read_text().splitlines() if line.strip()]
    if len(region_rows) != 1581 or any("bbox" in row or "target_bbox" in row for row in region_rows):
        raise ValueError("G2 scoring requires complete label-free region manifest")
    if args.limit is not None:
        region_rows = region_rows[:args.limit]
    roster = yaml.safe_load(ROSTER_PATH.read_text())
    spec = roster["models"][args.model]
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model_index_hash = sha256_file(args.model_dir / "model.safetensors.index.json")
    protocol_hash = canonical_hash(spec)
    engine = LLM(
        model=str(args.model_dir), tensor_parallel_size=8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=16384, max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": spec["min_pixels"], "max_pixels": spec["max_pixels"]},
        trust_remote_code=True, enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=spec["max_new_tokens"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    pending = [row for row in region_rows if row["id"] not in completed]
    written = 0
    with args.output.open("a", buffering=1) as output:
        for source in pending:
            image = Image.open(DATA_ROOT / "images" / source["img_filename"]).convert("RGB")
            by_index = {region["region_index"]: region for region in source["regions"]}
            required = source["required_region_indices_by_model"][args.model]
            requests, metadata = [], []
            for region_index in required:
                region = by_index[region_index]
                left, top, right, bottom = region["region"]
                crop = image if region_index == 0 else image.crop((left, top, right, bottom))
                prompt, resized, resized_size = prompt_for(args.model, spec, processor, crop, source["instruction"])
                requests.append({"prompt": prompt, "multi_modal_data": {"image": resized}})
                metadata.append((region, crop.size, resized_size))
            predictions = []
            for start in range(0, len(requests), args.batch_size):
                generated = engine.generate(requests[start:start + args.batch_size], sampling, use_tqdm=False)
                batch_metadata = metadata[start:start + args.batch_size]
                for result, (region, crop_size, resized_size) in zip(generated, batch_metadata):
                    response = result.outputs[0].text
                    point, parse_ok = parse_response(response, args.model, crop_size, resized_size)
                    left, top, _, _ = region["region"]
                    if region["region_index"] != 0:
                        point = [point[0] + left, point[1] + top] if parse_ok else [0.0, 0.0]
                    predictions.append({
                        "region_index": region["region_index"],
                        "region": region["region"],
                        "point": point,
                        "parse_ok": parse_ok,
                        "response": response,
                        "resized_size": resized_size,
                    })
            if [value["region_index"] for value in predictions] != required:
                raise ValueError(f"G2 score order mismatch: {args.model}/{source['id']}")
            artifact = {
                "stable_index": source["stable_index"], "id": source["id"],
                "application": source["application"], "img_filename": source["img_filename"],
                "img_size": source["img_size"], "instruction": source["instruction"],
                "model_id": args.model, "model_revision": spec["revision"],
                "model_index_sha256": model_index_hash, "protocol_sha256": protocol_hash,
                "region_manifest_sha256": source["regions_sha256"], "predictions": predictions,
            }
            artifact["predictions_sha256"] = canonical_hash(predictions)
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush(); os.fsync(output.fileno())
            written += 1
            print(json.dumps({"model": args.model, "written": written}), flush=True)
    print(json.dumps({"status": "PASS", "model": args.model, "written": written}), flush=True)


if __name__ == "__main__":
    main()
