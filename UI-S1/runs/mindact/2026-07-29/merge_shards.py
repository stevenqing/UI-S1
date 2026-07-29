import argparse
import collections
import json
import pickle
import sys
from pathlib import Path


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
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
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
    predictions_by_index = {}
    outputs_by_index = {}
    shard_provenance = []
    for shard_index in range(args.num_shards):
        shard = args.shard_root / f"shard-{shard_index}"
        provenance = json.loads((shard / "provenance.json").read_text())
        predictions = json.loads((shard / "test_task_predictions_top50.json").read_text())
        outputs = json.loads((shard / "test_task_outputs_top50.json").read_text())
        indices = provenance["global_indices"]
        if provenance["shard_index"] != shard_index or provenance["num_shards"] != args.num_shards:
            raise ValueError(f"shard configuration mismatch: {shard_index}")
        if len(indices) != len(predictions) or len(indices) != len(outputs):
            raise ValueError(f"shard coverage mismatch: {shard_index}")
        for index, prediction, output in zip(indices, predictions, outputs):
            if index in predictions_by_index:
                raise ValueError(f"duplicate global index {index}")
            predictions_by_index[index] = prediction
            outputs_by_index[index] = output
        shard_provenance.append(provenance)
    if set(predictions_by_index) != set(range(2094)):
        raise ValueError("merged coverage is not exactly 0..2093")

    predictions = [predictions_by_index[index] for index in range(2094)]
    outputs = [outputs_by_index[index] for index in range(2094)]
    episode_element = collections.defaultdict(list)
    episode_action = collections.defaultdict(list)
    episode_step = collections.defaultdict(list)
    for source, prediction in zip(data, predictions):
        identity = f"{source['annotation_id']}_{source['action_uid']}"
        if prediction[0] != identity:
            raise ValueError(f"identity mismatch: {identity}")
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
    results = {
        "element_acc": mean(flat_element),
        "action_f1": mean(flat_action),
        "step_acc": mean(flat_step),
        "marco_element_acc": mean([mean(values) for values in episode_element.values()]),
        "marco_action_f1": mean([mean(values) for values in episode_action.values()]),
        "marco_step_acc": mean([mean(values) for values in episode_step.values()]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_task_predictions_top50.json").write_text(json.dumps(predictions))
    (args.output_dir / "test_task_outputs_top50.json").write_text(json.dumps(outputs))
    (args.output_dir / "test_task_results_top50.json").write_text(json.dumps(results, indent=2) + "\n")
    aggregate = dict(shard_provenance[0])
    aggregate.update({
        "status": "COMPLETE",
        "actions": 2094,
        "episodes": 252,
        "shard_index": None,
        "global_indices": list(range(2094)),
        "result": results,
    })
    (args.output_dir / "provenance.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "actions": 2094, "result": results}, indent=2))


if __name__ == "__main__":
    main()
