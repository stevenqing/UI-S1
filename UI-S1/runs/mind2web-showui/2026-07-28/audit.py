import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_MODEL_REVISION = "cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60"
EXPECTED_PROCESSOR_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_answer(row: dict) -> dict:
    step = row["step"]
    action = step["operation"]["op"]
    if action == "TYPE":
        value = step["operation"]["value"]
    else:
        import re

        match = re.search(r"\]\s+(.*?)\s+->", row["step_repr"])
        value = match.group(1) if match else None
    bbox = step["bbox"]
    width, height = row["img_size"]
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    with args.predictions.open() as handle:
        predictions = [json.loads(line) for line in handle]
    score = json.loads(args.score.read_text())
    if len(metadata) != 2080 or len(predictions) != 2080:
        raise ValueError(
            f"coverage mismatch metadata={len(metadata)} predictions={len(predictions)}"
        )

    identities = set()
    for index, (source, prediction) in enumerate(zip(metadata, predictions)):
        if prediction["index"] != index:
            raise ValueError(f"index mismatch at row {index}")
        identity = (prediction["annot_id"], prediction["action_uid"])
        expected_identity = (source["annot_id"], source["action_uid"])
        if identity != expected_identity:
            raise ValueError(f"identity mismatch at row {index}")
        if identity in identities:
            raise ValueError(f"duplicate identity {identity}")
        identities.add(identity)
        if prediction["image"] != source["img_url"]:
            raise ValueError(f"image mismatch at row {index}")
        if prediction["answer"] != expected_answer(source):
            raise ValueError(f"answer mismatch at row {index}")
        if prediction["bbox"] != source["step"]["bbox"]:
            raise ValueError(f"bbox mismatch at row {index}")
        if prediction["image_size"] != source["img_size"]:
            raise ValueError(f"image size mismatch at row {index}")
        if prediction["model_revision"] != EXPECTED_MODEL_REVISION:
            raise ValueError(f"model revision mismatch at row {index}")
        if prediction["processor_revision"] != EXPECTED_PROCESSOR_REVISION:
            raise ValueError(f"processor revision mismatch at row {index}")
        if (
            prediction["num_history"],
            prediction["min_visual_tokens"],
            prediction["max_visual_tokens"],
            prediction["num_shards"],
        ) != (2, 256, 1344, 4):
            raise ValueError(f"configuration mismatch at row {index}")
        if not isinstance(prediction["response"], str):
            raise ValueError(f"non-string response at row {index}")
        if len(prediction["input_ids_sha256"]) != 64:
            raise ValueError(f"invalid input hash at row {index}")

    if score["rows"] != 2080 or score["episodes"] != 252:
        raise ValueError("score coverage does not match expected benchmark coverage")
    result = {
        "status": "PASS",
        "rows": len(predictions),
        "episodes": len({row["annot_id"] for row in predictions}),
        "unique_identities": len(identities),
        "predictions_sha256": sha256(args.predictions),
        "score_sha256": sha256(args.score),
        "metadata_sha256": sha256(args.metadata),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()