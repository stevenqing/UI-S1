import argparse
import hashlib
import json
import math
from pathlib import Path

from infer import (
    MAX_PIXELS,
    MODEL_NAME,
    MODEL_REVISION,
    SOURCE_REVISION,
    extract_action,
    extract_coordinate,
    format_prompt,
    resolve_image,
    sha256_bytes,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--prepared-images", type=Path, required=True)
    parser.add_argument("--recovered-images", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--gcs-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()

    source = json.loads(args.source_json.read_text())
    by_index = {}
    for shard_index in range(args.num_shards):
        path = args.shard_root / f"shard-{shard_index}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            index = row["index"]
            if index % args.num_shards != shard_index or row["shard_index"] != shard_index:
                raise ValueError(f"shard mismatch at index {index}")
            if index in by_index:
                raise ValueError(f"duplicate index {index}")
            by_index[index] = row
    if set(by_index) != set(range(7868)):
        raise ValueError("merged coverage is not exactly 0..7867")

    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    expected_shards = sorted(set(index["weight_map"].values()))
    if expected_shards != sorted(path.name for path in args.model_dir.glob("model-*.safetensors")):
        raise ValueError("model shard index mismatch")
    model_hashes = {
        name: sha256_file(args.model_dir / name) for name in expected_shards
    }
    gcs_manifest = json.loads(args.gcs_manifest.read_text())
    image_manifest = json.loads(args.image_manifest.read_text())
    if gcs_manifest["status"] != "DOWNLOADED_MD5_VERIFIED" or len(gcs_manifest["objects"]) != 22:
        raise ValueError("official GCS manifest is incomplete")
    if image_manifest["status"] != "COMPLETE" or image_manifest["total_image_coverage"] != 7868:
        raise ValueError("image extraction manifest is incomplete")

    type_correct = 0
    click_correct = 0
    click_total = 0
    merged = []
    for row_index, source_row in enumerate(source):
        prediction = by_index[row_index]
        if prediction["image"] != source_row["image"] or prediction["task"] != source_row["task"]:
            raise ValueError(f"source identity mismatch at row {row_index}")
        if prediction["gt"] != source_row["gt"]:
            raise ValueError(f"ground truth mismatch at row {row_index}")
        image_path = resolve_image(
            source_row["image"], args.prepared_images, args.recovered_images
        )
        if prediction["image_sha256"] != sha256_file(image_path):
            raise ValueError(f"image hash mismatch at row {row_index}")
        prompt = format_prompt(source_row["task"])
        if prediction["prompt_sha256"] != sha256_bytes(prompt.encode()):
            raise ValueError(f"prompt hash mismatch at row {row_index}")
        expected_contract = (
            MODEL_NAME,
            MODEL_REVISION,
            SOURCE_REVISION,
            "released_test_androidcontrol",
            "observed_672x1484_input_released_eval_ac_644x1484_scale",
            "temperature_0.1_top_k_1_top_p_0.001_repetition_1.05_max_1024",
            1,
            args.kv_cache_memory_bytes,
            args.num_shards,
            row_index % args.num_shards,
        )
        actual_contract = tuple(prediction[key] for key in (
            "model_name", "model_revision", "source_revision", "prompt_contract",
            "coordinate_contract", "generation", "tensor_parallel_size",
            "kv_cache_memory_bytes", "num_shards", "shard_index",
        ))
        if actual_contract != expected_contract:
            raise ValueError(f"configuration mismatch at row {row_index}")
        parsed_action = extract_action(prediction["response"])
        coordinate = extract_coordinate(prediction["response"])
        scaled = [int(coordinate[0] * 1080 / 644), int(coordinate[1] * 2400 / 1484)]
        if prediction["pred_action"] != parsed_action:
            raise ValueError(f"action parser mismatch at row {row_index}")
        if prediction["pred_coordinate_original"] != scaled:
            raise ValueError(f"coordinate parser mismatch at row {row_index}")
        if prediction["resized_image_size"] != [672, 1484]:
            raise ValueError(f"unexpected processor grid at row {row_index}")
        ground_truth_action = source_row["gt"]["action_type"]
        action_correct = parsed_action == ground_truth_action
        type_correct += int(action_correct)
        if ground_truth_action == "click":
            click_total += 1
            distance = math.dist(
                scaled,
                [source_row["gt"]["x"], source_row["gt"]["y"]],
            )
            click_correct += int(action_correct and distance < 1080 * 0.14)
        merged.append(prediction)

    type_accuracy = type_correct / len(source)
    grounding_accuracy = click_correct / click_total
    result = {
        "status": "PASS",
        "coverage": "COMPLETE",
        "rows": len(source),
        "episodes": len({row["image"].split("-screenshot_")[0] for row in source}),
        "type_accuracy": type_accuracy,
        "grounding_accuracy": grounding_accuracy,
        "reported_average": (type_accuracy + grounding_accuracy) / 2,
        "type_correct": type_correct,
        "type_total": len(source),
        "grounding_correct": click_correct,
        "grounding_total": click_total,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "max_pixels": MAX_PIXELS,
        "model_shard_sha256": model_hashes,
        "source_json_sha256": sha256_file(args.source_json),
        "image_manifest_sha256": sha256_file(args.image_manifest),
        "gcs_manifest_sha256": sha256_file(args.gcs_manifest),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in merged))
    result["predictions_sha256"] = sha256_file(predictions)
    (args.output_dir / "score_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()