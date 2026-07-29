import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor

from common import MODEL_REVISIONS, format_prompt, read_jsonl, sha256_file
from infer import GENERATION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=MODEL_REVISIONS, required=True)
    parser.add_argument("--setting", choices=("low", "high"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = read_jsonl(args.data)
    predictions = read_jsonl(args.predictions)
    score = json.loads(args.score.read_text())
    if len(sources) != 7708 or len(predictions) != 7708:
        raise ValueError("complete audit requires 7708 source and prediction rows")
    processor = AutoProcessor.from_pretrained(
        args.model_dir.resolve(),
        size={"shortest_edge": 3136, "longest_edge": 2116800},
        min_pixels=3136,
        max_pixels=2116800,
        use_fast=False,
    )
    identities = set()
    for index, (source, prediction) in enumerate(zip(sources, predictions)):
        if prediction["index"] != index:
            raise ValueError(f"index mismatch at row {index}")
        for key in ("identity", "episode_id", "step_id", "gt_action"):
            if prediction[key] != source[key]:
                raise ValueError(f"{key} mismatch at row {index}")
        if prediction["identity"] in identities:
            raise ValueError(f"duplicate identity at row {index}")
        identities.add(prediction["identity"])
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": format_prompt(source, args.setting)},
                {"type": "image"},
            ],
        }]
        serialized = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prediction["prompt_sha256"] != hashlib.sha256(serialized.encode()).hexdigest():
            raise ValueError(f"prompt hash mismatch at row {index}")
        expected = (
            args.model_name, MODEL_REVISIONS[args.model_name],
            "official_v1_mobile_use_with_androidcontrol_wait", "point_0_1000",
            GENERATION, args.setting, 4,
        )
        actual = tuple(prediction[key] for key in (
            "model_name", "model_revision", "prompt_contract", "coordinate_space",
            "generation", "setting", "num_shards",
        ))
        if actual != expected:
            raise ValueError(f"configuration mismatch at row {index}")
    if score["coverage"] != "COMPLETE" or score["rows"] != 7708:
        raise ValueError("score coverage mismatch")
    result = {
        "status": "PASS",
        "coverage": "COMPLETE",
        "rows": 7708,
        "episodes": len({row["episode_id"] for row in predictions}),
        "unique_identities": len(identities),
        "model_name": args.model_name,
        "model_revision": MODEL_REVISIONS[args.model_name],
        "setting": args.setting,
        "data_sha256": sha256_file(args.data),
        "predictions_sha256": sha256_file(args.predictions),
        "score_sha256": sha256_file(args.score),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()