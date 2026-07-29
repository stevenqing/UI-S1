import argparse
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_f1(prediction: str, reference: str) -> float:
    predicted = set(prediction.strip().split())
    expected = set(reference.strip().split())
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 0.0 if not overlap else 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str((args.source_dir / "action_prediction").resolve()))
    sys.path.insert(0, str(args.source_dir.resolve()))
    from dataloader import get_data_split

    with args.score_file.open("rb") as handle:
        candidate_results = pickle.load(handle)
    data = get_data_split(
        str(args.data_dir.resolve()),
        "test_task/*.json",
        candidate_results=candidate_results,
    )
    predictions = json.loads(args.predictions.read_text())
    reported = json.loads(args.results.read_text())
    provenance = json.loads(args.provenance.read_text())
    manifest = json.loads(args.manifest.read_text())
    if len(data) != 2094 or len(predictions) != 2094:
        raise ValueError("complete MindAct audit requires 2094 source and prediction rows")
    if provenance["status"] != "COMPLETE" or provenance["actions"] != 2094:
        raise ValueError("provenance is not complete")
    if manifest["status"] != "DOWNLOADED_HASH_VERIFIED":
        raise ValueError("artifact manifest is not verified")

    episode_element = collections.defaultdict(list)
    episode_action = collections.defaultdict(list)
    episode_step = collections.defaultdict(list)
    identities = set()
    for source, prediction in zip(data, predictions):
        identity = f"{source['annotation_id']}_{source['action_uid']}"
        if prediction[0] != identity or identity in identities:
            raise ValueError(f"identity/order mismatch: {identity}")
        identities.add(identity)
        positive_ids = {
            candidate["backend_node_id"]
            for candidate in source["pos_candidates"]
            if candidate["rank"] < 50
        }
        element = float(prediction[1] in positive_ids)
        reference_action = source["operation"]["op"]
        if reference_action != "CLICK":
            reference_action += " " + source["operation"]["value"]
        action = token_f1(prediction[2], reference_action)
        step = float(element == 1.0 and action == 1.0)
        episode_id = source["annotation_id"]
        episode_element[episode_id].append(element)
        episode_action[episode_id].append(action)
        episode_step[episode_id].append(step)

    flat_element = [value for values in episode_element.values() for value in values]
    flat_action = [value for values in episode_action.values() for value in values]
    flat_step = [value for values in episode_step.values() for value in values]
    recomputed = {
        "element_acc": mean(flat_element),
        "action_f1": mean(flat_action),
        "step_acc": mean(flat_step),
        "marco_element_acc": mean([mean(values) for values in episode_element.values()]),
        "marco_action_f1": mean([mean(values) for values in episode_action.values()]),
        "marco_step_acc": mean([mean(values) for values in episode_step.values()]),
    }
    for key, value in recomputed.items():
        if not np.isclose(value, reported[key], atol=1e-12):
            raise ValueError(f"metric mismatch for {key}: {value} != {reported[key]}")

    result = {
        "status": "PASS",
        "coverage": "COMPLETE",
        "actions": 2094,
        "episodes": len(episode_element),
        "unique_identities": len(identities),
        "metrics": recomputed,
        "predictions_sha256": sha256(args.predictions),
        "results_sha256": sha256(args.results),
        "provenance_sha256": sha256(args.provenance),
        "manifest_sha256": sha256(args.manifest),
        "candidate_scores_sha256": sha256(args.score_file),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()