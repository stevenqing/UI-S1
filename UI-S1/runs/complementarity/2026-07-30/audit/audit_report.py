import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {
    "solvable", "annotation_error", "partially_observable",
    "ambiguous_instruction", "evaluator_artifact",
}


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return None
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return [center - spread, center + spread]


def kappa(left, right):
    labels = sorted(LABELS)
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(left_counts[label] / len(left) * right_counts[label] / len(right) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = {row["sample_id"]: row for row in read_jsonl(args.sample)}
    labels = read_jsonl(args.labels)
    if any(row["label"] is None for row in labels):
        raise ValueError("human audit labels are incomplete; refusing to fabricate E4 results")
    if any(row["label"] not in LABELS for row in labels):
        raise ValueError("unknown E4 label")
    sample_ids = set(samples)
    assigned_ids = {row["sample_id"] for row in labels}
    expected_assigned_ids = {
        sample_id for sample_id, sample in samples.items()
        if sample["audit_stream"].startswith("e4/") and "cross_modal" not in sample["audit_stream"]
        or sample["audit_stream"] == "d2/select_visibility"
    }
    if assigned_ids != expected_assigned_ids:
        raise ValueError(
            f"audit assignment identity mismatch: missing={sorted(expected_assigned_ids - assigned_ids)[:10]} "
            f"extra={sorted(assigned_ids - expected_assigned_ids)[:10]}"
        )
    if not assigned_ids <= sample_ids:
        raise ValueError("audit labels contain unknown sample IDs")
    assignment_keys = [(row["sample_id"], row["annotator"]) for row in labels]
    if len(assignment_keys) != len(set(assignment_keys)):
        raise ValueError("duplicate sample/annotator assignment")
    by_sample = defaultdict(list)
    for row in labels:
        by_sample[row["sample_id"]].append(row)
    overlap = [rows for rows in by_sample.values() if len(rows) == 2]
    if len(overlap) != 30:
        raise ValueError(f"expected 30 double-annotated rows, found {len(overlap)}")
    agreement = kappa([rows[0]["label"] for rows in overlap], [rows[1]["label"] for rows in overlap])
    if agreement < 0.6:
        result = {"status": "REANNOTATION_REQUIRED", "cohen_kappa": agreement}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit("Cohen kappa below 0.6; align rubric and relabel")
    adjudicated = {sample_id: rows[0]["label"] for sample_id, rows in by_sample.items()}
    streams = defaultdict(list)
    for sample_id, label in adjudicated.items():
        streams[samples[sample_id]["audit_stream"]].append(label)
    report = {}
    for stream, values in streams.items():
        counts = Counter(values)
        report[stream] = {
            "rows": len(values),
            "labels": {
                label: {"count": counts[label], "rate": counts[label] / len(values), "wilson_95": wilson(counts[label], len(values))}
                for label in sorted(LABELS)
            },
            "annotation_plus_partial_rate": (counts["annotation_error"] + counts["partially_observable"]) / len(values),
            "tier4_curriculum": "CANCEL" if counts["annotation_error"] + counts["partially_observable"] > 0.4 * len(values) else "KEEP",
        }
    result = {"status": "PASS", "cohen_kappa": agreement, "streams": report}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "cohen_kappa": agreement}, indent=2))


if __name__ == "__main__":
    main()