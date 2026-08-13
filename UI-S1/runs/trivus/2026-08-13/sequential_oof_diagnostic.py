import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
PRIOR_DIR = RUN_DIR.parent / "2026-08-12"
OUTPUT_ROOT = RUN_DIR / "sequential_exploratory"
RESULT_PATH = RUN_DIR / "SEQUENTIAL_OOF_DIAGNOSTIC.json"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(PRIOR_DIR))

from context_common import atomic_json_file, sha256_file
from headroom_atlas import load_candidate_labels
from finalize_trivus import load_public
from sequential_oof_runner import FAMILIES, OOF_FIELDS


def ranking_metrics(rows, labels):
    if not rows:
        raise ValueError("Sequential OOF diagnostic has no rows")
    maximum = max(len(row["candidate_order"]) for row in rows)
    hits = defaultdict(list)
    reciprocal = []
    top1 = []
    oracle = []
    for row in rows:
        values = labels[row["sample_key"]]
        order = row["candidate_order"]
        if sorted(order) != list(range(len(values))):
            raise ValueError("Sequential OOF diagnostic order mismatch")
        ranked = [bool(values[index]) for index in order]
        positions = [index + 1 for index, success in enumerate(ranked) if success]
        top1.append(ranked[0])
        oracle.append(bool(positions))
        reciprocal.append(1.0 / positions[0] if positions else 0.0)
        for budget in range(1, maximum + 1):
            hits[budget].append(any(ranked[:budget]))
    oracle_rate = float(np.mean(oracle))
    return {
        "contexts": len(rows),
        "top1": float(np.mean(top1)),
        "mrr": float(np.mean(reciprocal)),
        "oracle": oracle_rate,
        "hit_at_k": {
            str(budget): float(np.mean(values))
            for budget, values in sorted(hits.items())
        },
        "oracle_recovery_at_k": {
            str(budget): float(np.sum(values) / np.sum(oracle)) if any(oracle) else 1.0
            for budget, values in sorted(hits.items())
        },
    }


def load_phase(phase, public):
    output = {family: [] for family in FAMILIES}
    root = OUTPUT_ROOT / phase
    for outer in range(5):
        for holdout in range(5):
            if holdout == outer:
                continue
            for family in FAMILIES:
                path = root / f"outer-{outer}" / f"holdout-{holdout}" / f"{family}.jsonl"
                rows = [
                    json.loads(line) for line in path.read_text().splitlines()
                    if line.strip()
                ]
                for row in rows:
                    if (
                        set(row) != OOF_FIELDS
                        or row["family"] != family
                        or row["fold"] != holdout
                        or row["sample_key"] not in public
                        or int(public[row["sample_key"]]["fold"]) != holdout
                    ):
                        raise ValueError(f"Sequential OOF diagnostic scope mismatch: {path}")
                output[family].extend(rows)
    for family, rows in output.items():
        contexts = [row["context_key"] for row in rows]
        if len(contexts) != len(set(contexts)):
            raise ValueError(f"Sequential duplicate OOF context: {family}")
    return output


def main():
    manifest_path = OUTPUT_ROOT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "PASS_EXPLORATORY_SEQUENTIAL_OOF_COMPLETE"
        or manifest.get("confirmatory") is not False
        or manifest.get("promotion_allowed") is not False
        or manifest.get("artifact_count") != 240
    ):
        raise PermissionError("Sequential OOF publication manifest mismatch")
    public = load_public()
    labels, label_manifests = load_candidate_labels(public)
    phases = {
        phase: load_phase(phase, public)
        for phase in ("cheap", "verifier")
    }
    metrics = {
        phase: {
            family: ranking_metrics(rows, labels)
            for family, rows in values.items()
        }
        for phase, values in phases.items()
    }
    result = {
        "schema_version": 1,
        "status": "PASS_EXPLORATORY_SEQUENTIAL_OOF_DIAGNOSTIC",
        "outcome": "EXPLORATORY_ONLY_NO_PROMOTION",
        "manifest_sha256": sha256_file(manifest_path),
        "label_manifests": label_manifests,
        "metrics": metrics,
        "verifier_minus_cheap": {
            family: {
                metric: metrics["verifier"][family][metric] - metrics["cheap"][family][metric]
                for metric in ("top1", "mrr")
            }
            for family in FAMILIES
        },
        "claim_boundary": {
            "confirmatory": False,
            "promotion_allowed": False,
            "repeated_sample_contexts": 4,
        },
    }
    if RESULT_PATH.exists():
        raise FileExistsError(RESULT_PATH)
    atomic_json_file(RESULT_PATH, result)
    print(json.dumps({
        phase: {
            family: {key: value[key] for key in ("top1", "mrr", "oracle")}
            for family, value in metrics[phase].items()
        }
        for phase in metrics
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()