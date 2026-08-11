import argparse
import json
from pathlib import Path

from evidence_data import MODES


RUN_DIR = Path(__file__).resolve().parent
VUS = RUN_DIR.parents[2] / "runs/visual-utility-selector/2026-08-11"


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    directory = RUN_DIR / f"evidence/{args.mode}"
    shards = sorted((directory / "raw").glob("shard-*.jsonl"))
    if len(shards) != 8:
        raise ValueError(f"RAVEL expected 8 shards: {len(shards)}")
    rows = []
    for shard in shards:
        rows.extend(load_jsonl(shard))
    by_key = {row["sample_key"]: row for row in rows}
    if len(rows) != 14644 or len(by_key) != len(rows):
        raise ValueError(f"RAVEL coverage mismatch: rows={len(rows)} unique={len(by_key)}")
    public = {
        row["sample_key"]: row
        for row in load_jsonl(VUS / "data/public_records.jsonl")
    }
    if set(public) != set(by_key):
        raise ValueError("RAVEL public/prediction identity mismatch")
    model_hashes = {row["model_index_sha256"] for row in rows}
    if len(model_hashes) != 1:
        raise ValueError(f"RAVEL model hash mismatch: {model_hashes}")
    ratios = []
    visual_tokens = []
    for key, row in by_key.items():
        if row["mode"] != args.mode:
            raise ValueError(f"RAVEL mode mismatch: {key}")
        if row["image_sha256"] != public[key]["image_sha256"]:
            raise ValueError(f"RAVEL image hash mismatch: {key}")
        if len(row["label_probabilities"]) != 12 or abs(sum(row["label_probabilities"]) - 1) > 1e-4:
            raise ValueError(f"RAVEL probability mismatch: {key}")
        budget = row["visual_budget"]
        ratio = float(budget["actual_pixel_ratio_vs_vus"])
        if not 0.90 <= ratio <= 1.02:
            raise ValueError(f"RAVEL-K1 budget ratio: {key}/{ratio}")
        if budget["actual_processed_pixels"] != budget["expected_total_processed_pixels"]:
            raise ValueError(f"RAVEL processor budget mismatch: {key}")
        ratios.append(ratio)
        visual_tokens.append(int(budget["actual_visual_tokens"]))
    output = directory / "predictions.jsonl"
    with output.open("w") as handle:
        for key in sorted(by_key):
            handle.write(json.dumps(by_key[key], ensure_ascii=True, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "PASS_BLIND_EVIDENCE_LOCKED",
        "mode": args.mode,
        "records": len(by_key),
        "shards": 8,
        "model_index_sha256": next(iter(model_hashes)),
        "predictions_sha256": sha256_file(output),
        "public_records_sha256": sha256_file(VUS / "data/public_records.jsonl"),
        "private_labels_opened": False,
        "pixel_ratio": {"min": min(ratios), "mean": sum(ratios) / len(ratios), "max": max(ratios)},
        "visual_tokens": {"min": min(visual_tokens), "mean": sum(visual_tokens) / len(visual_tokens), "max": max(visual_tokens)},
    }
    (directory / "predictions.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
