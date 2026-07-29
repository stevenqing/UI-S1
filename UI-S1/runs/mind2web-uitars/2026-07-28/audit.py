import argparse
import json
from pathlib import Path

from common import MODEL_REVISIONS, expected_answer, prompt_sha256, read_json, read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--model-name", choices=MODEL_REVISIONS, required=True)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = read_json(args.metadata)
    predictions = read_jsonl(args.predictions)
    score = json.loads(args.score.read_text())
    if len(metadata) != 2080 or len(predictions) != 2080:
        raise ValueError("complete audit requires 2080 metadata and prediction rows")
    identities = set()
    expected_config = (
        args.model_name,
        MODEL_REVISIONS[args.model_name],
        "official_v1_computer_use_single_round",
        "point_0_1000",
        "greedy_frequency_penalty_1",
        128,
        args.num_shards,
    )
    for index, prediction in enumerate(predictions):
        source = metadata[index]
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
        if prediction["prompt_sha256"] != prompt_sha256(source):
            raise ValueError(f"prompt hash mismatch at row {index}")
        actual_config = tuple(prediction[key] for key in (
            "model_name", "model_revision", "prompt_contract", "coordinate_space",
            "generation", "max_new_tokens", "num_shards",
        ))
        if actual_config != expected_config:
            raise ValueError(f"configuration mismatch at row {index}")
        if args.tensor_parallel_size is not None:
            expected_runtime = (
                args.tensor_parallel_size,
                args.kv_cache_memory_bytes,
                args.enforce_eager,
            )
            actual_runtime = tuple(prediction[key] for key in (
                "tensor_parallel_size", "kv_cache_memory_bytes", "enforce_eager",
            ))
            if actual_runtime != expected_runtime:
                raise ValueError(f"runtime configuration mismatch at row {index}")
    if score["rows"] != 2080 or score["episodes"] != 252:
        raise ValueError("score coverage mismatch")
    if args.checkpoint_manifest:
        manifest = json.loads(args.checkpoint_manifest.read_text())
        if manifest["status"] != "DOWNLOADED_HASH_INDEX_VERIFIED":
            raise ValueError("checkpoint manifest is not fully verified")
        if manifest["model"] != args.model_name or manifest["revision"] != MODEL_REVISIONS[args.model_name]:
            raise ValueError("checkpoint manifest identity mismatch")
    result = {
        "status": "PASS",
        "coverage": "COMPLETE",
        "rows": 2080,
        "episodes": 252,
        "unique_identities": len(identities),
        "model_name": args.model_name,
        "model_revision": MODEL_REVISIONS[args.model_name],
        "metadata_sha256": sha256_file(args.metadata),
        "predictions_sha256": sha256_file(args.predictions),
        "score_sha256": sha256_file(args.score),
    }
    if args.checkpoint_manifest:
        result["checkpoint_manifest_sha256"] = sha256_file(args.checkpoint_manifest)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()