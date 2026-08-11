import argparse
import json
from pathlib import Path

import yaml

from vus_data import CANDIDATE_LABELS, sha256_file


RUN_DIR = Path(__file__).resolve().parent


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=RUN_DIR / "data/public_records.jsonl")
    parser.add_argument("--raw-dir", type=Path, default=RUN_DIR / "zero_shot/raw")
    parser.add_argument("--output", type=Path, default=RUN_DIR / "zero_shot/predictions.jsonl")
    args = parser.parse_args()
    config = yaml.safe_load((RUN_DIR / "configs/vus_prereg.yaml").read_text())
    expected = {row["sample_key"]: row for row in load_jsonl(args.records)}
    shards = sorted(args.raw_dir.glob("shard-*.jsonl"))
    if len(shards) != config["eligibility_anchor"]["shards"]:
        raise ValueError(f"expected 8 zero-shot shards, found {len(shards)}")
    predictions = []
    for shard in shards:
        predictions.extend(load_jsonl(shard))
    by_key = {row["sample_key"]: row for row in predictions}
    if len(by_key) != len(predictions):
        raise ValueError("duplicate blind prediction keys")
    if set(by_key) != set(expected):
        raise ValueError(f"blind prediction coverage mismatch: predicted={len(by_key)} expected={len(expected)}")
    model_hashes = {row["model_index_sha256"] for row in predictions}
    if len(model_hashes) != 1:
        raise ValueError(f"model hash mismatch across shards: {model_hashes}")
    for key, prediction in by_key.items():
        source = expected[key]
        if prediction["image_sha256"] != source["image_sha256"]:
            raise ValueError(f"blind image hash mismatch: {key}")
        if len(prediction["label_probabilities"]) != len(CANDIDATE_LABELS):
            raise ValueError(f"label width mismatch: {key}")
        if abs(sum(prediction["label_probabilities"]) - 1.0) > 1e-4:
            raise ValueError(f"probability normalization mismatch: {key}")
        if sorted(prediction["display_to_candidate"]) != list(range(12)):
            raise ValueError(f"candidate permutation mismatch: {key}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for key in sorted(by_key):
            handle.write(json.dumps(by_key[key], ensure_ascii=True, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "PASS_BLIND_INFERENCE_COMPLETE",
        "records": len(by_key),
        "shards": len(shards),
        "model_index_sha256": next(iter(model_hashes)),
        "public_records_sha256": sha256_file(args.records),
        "predictions_sha256": sha256_file(args.output),
        "private_labels_opened": False,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
