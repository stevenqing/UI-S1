import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


THRESHOLD = 0.2469955724225174


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def load_generated(shard_root, num_shards):
    rows = {}
    for shard in range(num_shards):
        path = shard_root / f"shard-{shard}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows or row["shard_index"] != shard:
                raise ValueError("generated eligibility identity mismatch")
            rows[row["id"]] = row
    if len(rows) != 1581:
        raise ValueError(f"eligibility requires 1,581 rows, found {len(rows)}")
    return [rows[key] for key in sorted(rows)]


def load_qwen3(path, annotations):
    source = json.loads(path.read_text())
    grouped = defaultdict(list)
    for row in source["details"]:
        key = (Path(row["img_path"]).name, tuple(row["bbox"]), row["prompt_to_evaluate"])
        grouped[key].append(row)
    selected = []
    for annotation in annotations:
        key = (Path(annotation["img_filename"]).name, tuple(annotation["bbox"]), annotation["instruction"])
        matches = grouped[key]
        if not matches:
            raise ValueError(f"Qwen3 eligibility missing identity: {key}")
        row = matches[0]
        predictions = row.get("_debug", {}).get("predictions", [])
        point = predictions[0].get("point") if predictions else None
        selected.append({
            "id": annotation["id"],
            "application": annotation["application"],
            "target_bbox": annotation["bbox"],
            "point": point if isinstance(point, list) and len(point) >= 2 else None,
            "parse_ok": isinstance(point, list) and len(point) >= 2,
        })
    return selected


def annotations(root):
    rows = []
    for path in sorted(root.glob("*.json")):
        rows.extend(json.loads(path.read_text()))
    rows.sort(key=lambda row: row["id"])
    if len(rows) != 1581 or len({row["id"] for row in rows}) != 1581:
        raise ValueError("ScreenSpot annotation coverage mismatch")
    return rows


def summarize(model_id, revision, rows, source_sha256):
    labels = []
    parsed = 0
    for row in rows:
        point = row.get("point")
        parse_ok = point is not None and len(point) >= 2
        parsed += int(parse_ok)
        labels.append(bool(parse_ok and point_in_bbox(point, row["target_bbox"])))
    correct = sum(labels)
    accuracy = correct / len(rows)
    return {
        "model_id": model_id,
        "model_revision": revision,
        "rows": len(rows),
        "parse_successes": parsed,
        "parse_rate": parsed / len(rows),
        "correct": correct,
        "bare_accuracy": accuracy,
        "minimum_bare_accuracy": THRESHOLD,
        "eligible": accuracy >= THRESHOLD,
        "source_sha256": source_sha256,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--generated-shards", type=Path)
    parser.add_argument("--qwen3-trace", type=Path)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotation_rows = annotations(args.annotation_root)
    if (args.generated_shards is None) == (args.qwen3_trace is None):
        raise ValueError("provide exactly one eligibility source")
    if args.generated_shards is not None:
        generated = load_generated(args.generated_shards, args.num_shards)
        by_id = {row["id"]: row for row in generated}
        rows = []
        for annotation in annotation_rows:
            row = by_id[annotation["id"]]
            prediction = row["predictions"][0]
            rows.append({
                "id": row["id"], "application": row["application"],
                "target_bbox": row["target_bbox"], "point": prediction["point"],
            })
        source_hash = hashlib.sha256(
            "".join(sorted(row["prediction_sha256"] for row in generated)).encode()
        ).hexdigest()
    else:
        rows = load_qwen3(args.qwen3_trace, annotation_rows)
        source_hash = hashlib.sha256(args.qwen3_trace.read_bytes()).hexdigest()
    result = {"status": "PASS", **summarize(args.model_id, args.model_revision, rows, source_hash)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
