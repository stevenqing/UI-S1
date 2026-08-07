import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1_qwen import infer, load_model, prompt_text, sha256_file


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate crop inference ids")
    return set(ids)


def load_sources(path):
    rows = {}
    for source_path in sorted(path.glob("shard-*.jsonl")) if path.is_dir() else [path]:
        for line in source_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate crop-source id: {row['id']}")
            rows[row["id"]] = row
    if len(rows) != 2080:
        raise ValueError(f"crop source requires 2,080 rows, found {len(rows)}")
    return rows


def regions_for(source, set_name):
    if set_name.startswith("view"):
        index = int(set_name.removeprefix("view"))
        return [source["regions"][index]["region"]]
    return source["arms"][set_name]


def source_hash(source, set_name):
    if set_name.startswith("view"):
        return source["regions_sha256"]
    return source["arms_sha256"]


def remap_prediction(prediction, region, image_size):
    if not prediction.get("parse_ok") or prediction.get("position") is None:
        return prediction
    left, top, right, bottom = region
    width, height = image_size
    crop_x, crop_y = prediction["position"]
    value = dict(prediction)
    value["crop_position"] = [crop_x, crop_y]
    value["position"] = [
        (left + crop_x * (right - left)) / width,
        (top + crop_y * (bottom - top)) / height,
    ]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=("tongui", "uitars"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--sets", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    sets = args.sets.split(",")
    if len(sets) != len(set(sets)) or not sets:
        raise ValueError("invalid crop set list")
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    model_spec = next(model for model in roster["mind2web"]["models"] if model["id"] == args.model_id)
    if (ROOT / model_spec["local_path"]).resolve() != args.model_dir.resolve():
        raise ValueError("model path differs from frozen roster")
    canonical_rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    sources = load_sources(args.regions)
    if set(sources) != {row["id"] for row in canonical_rows}:
        raise ValueError("crop source identity mismatch")
    indices = list(range(args.shard_index, len(canonical_rows), args.num_shards))
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
            row = canonical_rows[index]
            if row["id"] in completed:
                continue
            source = sources[row["id"]]
            image = Image.open(ROOT / row["image"]).convert("RGB")
            predictions = {}
            for set_name in sets:
                predictions[set_name] = []
                for crop_index, region in enumerate(regions_for(source, set_name)):
                    crop = image.crop(region)
                    response, prediction = infer(crop, prompt_text(roster, row), processor, model, args.model_type)
                    predictions[set_name].append({
                        "crop_index": crop_index,
                        "region": region,
                        "response": response,
                        "prediction": remap_prediction(prediction, region, image.size),
                    })
            artifact = {
                "stable_index": index,
                "id": row["id"],
                "image_sha256": row["image_sha256"],
                "model_id": args.model_id,
                "model_revision": model_spec["revision"],
                "model_index_sha256": index_hash,
                "sets": sets,
                "source_hashes": {set_name: source_hash(source, set_name) for set_name in sets},
                "predictions": predictions,
                "predictions_sha256": hashlib.sha256(json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    print(json.dumps({
        "status": "PASS", "model": args.model_id, "shard": args.shard_index,
        "completed": len(completed_ids(args.output)),
    }))


if __name__ == "__main__":
    main()