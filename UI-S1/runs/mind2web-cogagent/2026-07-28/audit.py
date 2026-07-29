import argparse
import json
from pathlib import Path

from common import (
    MODEL_NAME,
    MODEL_REVISION,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    expected_answer,
    prompt_sha256,
    read_json,
    read_jsonl,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    metadata = read_json(args.metadata)
    predictions = read_jsonl(args.predictions)
    score = json.loads(args.score.read_text())
    if args.require_complete and (len(metadata) != 2080 or len(predictions) != 2080):
        raise ValueError("complete audit requires 2080 metadata and prediction rows")

    identities = set()
    for row_number, prediction in enumerate(predictions):
        index = prediction["index"]
        if index != row_number:
            raise ValueError(f"index/order mismatch at row {row_number}")
        source = metadata[index]
        identity = (source["annot_id"], source["action_uid"])
        if identity in identities:
            raise ValueError(f"duplicate identity {identity}")
        identities.add(identity)
        if (prediction["annot_id"], prediction["action_uid"]) != identity:
            raise ValueError(f"identity mismatch at row {index}")
        if prediction["image"] != source["img_url"]:
            raise ValueError(f"image mismatch at row {index}")
        if prediction["bbox"] != source["step"]["bbox"]:
            raise ValueError(f"bbox mismatch at row {index}")
        if prediction["image_size"] != source["img_size"]:
            raise ValueError(f"image size mismatch at row {index}")
        if prediction["answer"] != expected_answer(source):
            raise ValueError(f"answer mismatch at row {index}")
        if prediction["prompt_sha256"] != prompt_sha256(source):
            raise ValueError(f"prompt hash mismatch at row {index}")
        expected_config = (
            MODEL_NAME,
            MODEL_REVISION,
            TOKENIZER_NAME,
            TOKENIZER_REVISION,
            "official_generic_single_round_with_grounding",
            "bbox_0_1000",
            "greedy",
        )
        actual_config = tuple(
            prediction[key]
            for key in (
                "model_name",
                "model_revision",
                "tokenizer_name",
                "tokenizer_revision",
                "prompt_contract",
                "coordinate_space",
                "generation",
            )
        )
        if actual_config != expected_config:
            raise ValueError(f"configuration mismatch at row {index}")
    if score["rows"] != len(predictions) or score["episodes"] != len(
        {row["annot_id"] for row in predictions}
    ):
        raise ValueError("score coverage mismatch")

    result = {
        "status": "PASS",
        "coverage": "COMPLETE" if len(predictions) == 2080 else "PARTIAL",
        "rows": len(predictions),
        "episodes": score["episodes"],
        "unique_identities": len(identities),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "metadata_sha256": sha256_file(args.metadata),
        "predictions_sha256": sha256_file(args.predictions),
        "score_sha256": sha256_file(args.score),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()