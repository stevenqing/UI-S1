import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_answer(source: dict) -> dict:
    step = source["step"]
    action = step["operation"]["op"]
    if action == "TYPE":
        value = step["operation"]["value"]
    else:
        match = re.search(r"\]\s+(.*?)\s+->", source["step_repr"])
        value = match.group(1) if match else None
    bbox = step["bbox"]
    width, height = source["img_size"]
    position = [
        round((bbox["x"] + bbox["width"] / 2) / width, 2),
        round((bbox["y"] + bbox["height"] / 2) / height, 2),
    ]
    return {"action": action, "value": value, "position": position}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--input-hash-reference", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    predictions = read_jsonl(args.predictions)
    references = read_jsonl(args.input_hash_reference)
    score = json.loads(args.score.read_text())
    manifest = json.loads(args.checkpoint_manifest.read_text())
    if len(metadata) != 2080 or len(predictions) != 2080 or len(references) != 2080:
        raise ValueError("complete audit requires 2080 metadata, prediction, and reference rows")
    reference_hashes = {row["index"]: row["input_ids_sha256"] for row in references}
    if set(reference_hashes) != set(range(2080)):
        raise ValueError("input hash reference coverage mismatch")
    if manifest["status"] != "DOWNLOADED_HASH_VERIFIED":
        raise ValueError("checkpoint manifest is not verified")

    identities = set()
    expected_config = (
        args.model_name,
        args.model_revision,
        "v2",
        2,
        "vtvt",
        256,
        1344,
        "vllm-0.11.0-tp4-native-lora",
        "greedy",
        1,
    )
    for index, (source, prediction) in enumerate(zip(metadata, predictions)):
        identity = (source["annot_id"], source["action_uid"])
        if prediction["index"] != index or (prediction["annot_id"], prediction["action_uid"]) != identity:
            raise ValueError(f"identity mismatch at row {index}")
        if identity in identities:
            raise ValueError(f"duplicate identity {identity}")
        identities.add(identity)
        if prediction["image"] != source["img_url"]:
            raise ValueError(f"image mismatch at row {index}")
        if prediction["bbox"] != source["step"]["bbox"] or prediction["image_size"] != source["img_size"]:
            raise ValueError(f"reference mismatch at row {index}")
        if prediction["answer"] != expected_answer(source):
            raise ValueError(f"answer mismatch at row {index}")
        if prediction["input_ids_sha256"] != reference_hashes[index]:
            raise ValueError(f"input tensor hash mismatch at row {index}")
        actual_config = tuple(prediction[key] for key in (
            "model_name", "model_revision", "version", "num_history",
            "interleaved_history", "min_visual_tokens", "max_visual_tokens",
            "attention_backend", "generation", "num_shards",
        ))
        if actual_config != expected_config:
            raise ValueError(f"configuration mismatch at row {index}")
    if score["rows"] != 2080 or score["episodes"] != 252:
        raise ValueError("score coverage mismatch")

    result = {
        "status": "PASS",
        "coverage": "COMPLETE",
        "rows": 2080,
        "episodes": 252,
        "unique_identities": len(identities),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "base_revision": manifest["base"]["revision"],
        "adapter_revision": manifest["adapter"]["revision"],
        "metadata_sha256": sha256(args.metadata),
        "predictions_sha256": sha256(args.predictions),
        "score_sha256": sha256(args.score),
        "input_hash_reference_sha256": sha256(args.input_hash_reference),
        "checkpoint_manifest_sha256": sha256(args.checkpoint_manifest),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()