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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input-hash-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    with args.predictions.open() as handle:
        predictions = [json.loads(line) for line in handle]
    score = json.loads(args.score.read_text())
    if len(metadata) != 2080 or len(predictions) != 2080:
        raise ValueError("coverage mismatch")
    reference_hashes = None
    if args.input_hash_reference:
        with args.input_hash_reference.open() as handle:
            reference_rows = [json.loads(line) for line in handle]
        reference_hashes = {row["index"]: row["input_ids_sha256"] for row in reference_rows}
        if set(reference_hashes) != set(range(2080)) or len(reference_rows) != 2080:
            raise ValueError("input hash reference coverage mismatch")
    identities = set()
    for index, (source, prediction) in enumerate(zip(metadata, predictions)):
        identity = (source["annot_id"], source["action_uid"])
        if prediction["index"] != index or (prediction["annot_id"], prediction["action_uid"]) != identity:
            raise ValueError(f"identity mismatch at row {index}")
        if identity in identities:
            raise ValueError(f"duplicate identity {identity}")
        identities.add(identity)
        if prediction["image"] != source["img_url"] or prediction["bbox"] != source["step"]["bbox"] or prediction["image_size"] != source["img_size"]:
            raise ValueError(f"reference mismatch at row {index}")
        if prediction["answer"] != expected_answer(source):
            raise ValueError(f"ground-truth answer mismatch at row {index}")
        if prediction["model_name"] != args.model_name or prediction["model_revision"] != args.model_revision:
            raise ValueError(f"model mismatch at row {index}")
        if reference_hashes is not None and prediction["input_ids_sha256"] != reference_hashes[index]:
            raise ValueError(f"input tensor hash mismatch at row {index}")
        expected_config = ("v2", 2, "vtvt", 256, 1344, "flash_attention_2", "greedy", 4)
        actual_config = tuple(prediction[key] for key in ("version", "num_history", "interleaved_history", "min_visual_tokens", "max_visual_tokens", "attention_backend", "generation", "num_shards"))
        if actual_config != expected_config:
            raise ValueError(f"configuration mismatch at row {index}")
    if score["rows"] != 2080 or score["episodes"] != 252:
        raise ValueError("score coverage mismatch")
    result = {
        "status": "PASS", "model_name": args.model_name, "model_revision": args.model_revision,
        "rows": 2080, "episodes": 252, "unique_identities": len(identities),
        "predictions_sha256": sha256(args.predictions), "score_sha256": sha256(args.score),
        "metadata_sha256": sha256(args.metadata),
    }
    if args.input_hash_reference:
        result["input_hash_reference_sha256"] = sha256(args.input_hash_reference)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()