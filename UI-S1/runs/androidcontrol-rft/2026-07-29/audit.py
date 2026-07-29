import argparse
import hashlib
import json
import math
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as parquet
from PIL import Image

from infer import (
    DATASET_REVISION,
    MODEL_REVISIONS,
    MODEL_TO_GT_ACTION_MAP,
    OFFICIAL_REPO_REVISION,
    canonical_sha256,
    load_official_parsers,
    load_prompt_template,
    sha256_bytes,
    source_provenance,
)


GROUNDING_ACTIONS = {"click", "long_press", "moveto", "doubleclick", "rightclick"}
TEXT_ACTIONS = {"type", "open_app", "scroll", "select"}
SIMPLE_ACTIONS = {
    "press_back", "wait", "navigate_back", "press_home", "complete", "impossible",
    "press_space", "press_enter", "press_down", "hotkey", "press_tab", "press_pgdn",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_f1(prediction, reference) -> float:
    if not isinstance(prediction, str) or not isinstance(reference, str):
        return 0.0
    predicted_tokens = set(prediction.lower().split())
    reference_tokens = set(reference.lower().split())
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0
    overlap = len(predicted_tokens & reference_tokens)
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def update_metrics(totals, row) -> None:
    action_correct = row["pred_action"] == row["gt_action"]
    totals["action"]["total"] += 1
    totals["action"]["correct"] += int(action_correct)
    step_success = False
    if row["gt_action"] in GROUNDING_ACTIONS:
        totals["grounding"]["total"] += 1
        predicted_x, predicted_y = row["pred_coord"][:2]
        ground_truth_x, ground_truth_y = row["gt_bbox"][:2]
        width, height = row["image_size"]
        distance_squared = (
            ((ground_truth_x - predicted_x) / width) ** 2
            + ((ground_truth_y - predicted_y) / height) ** 2
        )
        grounding_correct = distance_squared < 0.14**2
        totals["grounding"]["correct"] += int(grounding_correct)
        step_success = action_correct and grounding_correct
    elif row["gt_action"] in TEXT_ACTIONS:
        totals["text"]["total"] += 1
        text_correct = text_f1(row["pred_input_text"], row["gt_input_text"]) >= 0.5
        totals["text"]["correct"] += int(text_correct)
        step_success = action_correct and text_correct
    elif row["gt_action"] in SIMPLE_ACTIONS:
        step_success = action_correct
    totals["step_success"]["total"] += 1
    totals["step_success"]["correct"] += int(step_success)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--data-setting", choices=("low", "high"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-name", choices=MODEL_REVISIONS, required=True)
    parser.add_argument("--prompt-template", required=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_rows = parquet.read_table(args.data_path).to_pylist()
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    score = json.loads(args.score.read_text())
    manifest = json.loads(args.manifest.read_text())
    if len(source_rows) != 7708 or len(predictions) != 7708:
        raise ValueError("complete audit requires 7708 source and prediction rows")
    if manifest["status"] != "DOWNLOADED_HASH_INDEX_VERIFIED":
        raise ValueError("artifact manifest status is not verified")
    if manifest["data_sha256"][args.data_setting] != sha256_file(args.data_path):
        raise ValueError("dataset hash mismatch")
    model_manifest = manifest["models"][args.model_name]
    if model_manifest["revision"] != MODEL_REVISIONS[args.model_name]:
        raise ValueError("model revision mismatch")

    prompt_template = load_prompt_template(args.prompt_template)
    extract_action, extract_coordinates, extract_parameter, extract_gui_r1_parameter = (
        load_official_parsers()
    )
    totals = defaultdict(lambda: defaultdict(int))
    for index, (source, prediction) in enumerate(zip(source_rows, predictions)):
        if prediction["index"] != index:
            raise ValueError(f"ordered identity mismatch at row {index}")
        expected_fields = {
            "data_setting": args.data_setting,
            "instruction": source["instruction"],
            "history": source.get("history"),
            "gt_action": source["gt_action"],
            "gt_bbox": source["gt_bbox"],
            "gt_input_text": source["gt_input_text"].lower()
            if source["gt_action"] == "scroll" else source["gt_input_text"],
            "group": source["group"],
            "ui_type": source["ui_type"],
            "model_name": args.model_name,
            "model_revision": MODEL_REVISIONS[args.model_name],
            "dataset_revision": DATASET_REVISION,
            "official_repo_revision": OFFICIAL_REPO_REVISION,
            "prompt_template": args.prompt_template,
            "image_processor": "slow_transformers_4_52_compatible",
            "generation": "temperature_0_max_tokens_256_skip_special_tokens_false",
            "tensor_parallel_size": 1,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "enforce_eager": True,
            "num_shards": args.num_shards,
            "shard_index": index % args.num_shards,
        }
        for key, expected in expected_fields.items():
            if prediction[key] != expected:
                raise ValueError(f"{key} mismatch at row {index}")
        image_sha256, source_sha256 = source_provenance(source)
        if prediction["image_sha256"] != image_sha256 or prediction["source_sha256"] != source_sha256:
            raise ValueError(f"source hash mismatch at row {index}")
        image_size = list(Image.open(BytesIO(source["image"]["bytes"])).size)
        if prediction["image_size"] != image_size:
            raise ValueError(f"image size mismatch at row {index}")
        text_prompt = prompt_template.format(
            instruction=source["instruction"], history=source.get("history", "None")
        )
        if prediction["text_prompt_sha256"] != sha256_bytes(text_prompt.encode()):
            raise ValueError(f"prompt hash mismatch at row {index}")
        predicted_action_raw = extract_action(prediction["pred_raw"])
        predicted_action = MODEL_TO_GT_ACTION_MAP.get(predicted_action_raw, predicted_action_raw)
        if prediction["pred_action"] != predicted_action:
            raise ValueError(f"parsed action mismatch at row {index}")
        coordinate, _, _ = extract_coordinates(prediction["pred_raw"])
        if coordinate is None:
            coordinate = [0, 0]
        expected_coordinate = [
            coordinate[0] * prediction["resize_scale"][0],
            coordinate[1] * prediction["resize_scale"][1],
        ]
        if not all(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9)
                   for actual, expected in zip(prediction["pred_coord"], expected_coordinate)):
            raise ValueError(f"parsed coordinate mismatch at row {index}")
        predicted_parameter = extract_parameter(prediction["pred_raw"])
        if args.prompt_template == "gui_r1":
            predicted_parameter = extract_gui_r1_parameter(prediction["pred_raw"])
        if prediction["pred_input_text"] != predicted_parameter:
            raise ValueError(f"parsed parameter mismatch at row {index}")
        update_metrics(totals, prediction)

    for metric in ("action", "grounding", "text", "step_success"):
        for count in ("correct", "total"):
            if score["metrics"][metric][count] != totals[metric][count]:
                raise ValueError(f"score mismatch for {metric}.{count}")
    result = {
        "status": "PASS",
        "coverage": "COMPLETE",
        "rows": 7708,
        "unique_ordered_indices": 7708,
        "model_name": args.model_name,
        "model_revision": MODEL_REVISIONS[args.model_name],
        "data_setting": args.data_setting,
        "data_sha256": sha256_file(args.data_path),
        "predictions_sha256": sha256_file(args.predictions),
        "score_sha256": sha256_file(args.score),
        "manifest_sha256": sha256_file(args.manifest),
        "metric_counts_sha256": canonical_sha256({
            metric: dict(counts) for metric, counts in totals.items()
        }),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()