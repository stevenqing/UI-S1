import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scoring import label_android_row


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    if args.require_complete and (len(rows) != 7708 or [row["index"] for row in rows] != list(range(7708))):
        raise ValueError("complete W2 AndroidControl scoring requires ordered indices 0..7707")
    labels = [label_android_row(row) for row in rows]
    result = {
        "status": "PASS", "coverage": "COMPLETE" if len(rows) == 7708 else "PARTIAL",
        "rows": len(rows), "model": rows[0]["model"] if rows else None,
        "setting": rows[0]["data_setting"] if rows else None,
        "view_id": rows[0]["view_id"] if rows else None,
        "step_successes": sum(label["step_success"] for label in labels),
        "step_sr": sum(label["step_success"] for label in labels) / len(labels),
        "action_accuracy": sum(label["action_correct"] for label in labels) / len(labels),
        "error_counts": dict(sorted(Counter(label["error_type"] for label in labels).items())),
        "predictions_sha256": sha256_file(args.predictions),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()