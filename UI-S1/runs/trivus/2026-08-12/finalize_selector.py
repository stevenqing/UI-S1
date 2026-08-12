import hashlib
import json
import math
from pathlib import Path

from selector_data import (
    RUN_DIR, assert_selector_environment, atomic_json, build_prompt, load_config,
    load_jsonl, render_overlay, rendered_image_sha256, sha256_file, write_jsonl,
)


def validate_prediction(row, public, config, verify_overlay=True):
    expected_fields = {
        "schema_version", "sample_key", "benchmark", "setting", "row_id", "fold", "group",
        "display_to_candidate", "selected_label", "selected_candidate_index", "label_logits",
        "label_probabilities", "prompt_sha256", "overlay_sha256", "image_sha256",
        "model_index_sha256",
    }
    if set(row) != expected_fields:
        raise ValueError(f"TriVUS selector prediction schema mismatch: {set(row) ^ expected_fields}")
    if sorted(row["display_to_candidate"]) != [0, 1, 2]:
        raise ValueError("TriVUS selector display permutation mismatch")
    if len(row["label_logits"]) != 3 or len(row["label_probabilities"]) != 3:
        raise ValueError("TriVUS selector logit width mismatch")
    if not all(math.isfinite(float(value)) for value in row["label_logits"] + row["label_probabilities"]):
        raise ValueError("TriVUS selector non-finite values")
    if not all(0.0 <= float(value) <= 1.0 for value in row["label_probabilities"]):
        raise ValueError("TriVUS selector probability range mismatch")
    if not math.isclose(sum(row["label_probabilities"]), 1.0, abs_tol=1e-6):
        raise ValueError("TriVUS selector probability sum mismatch")
    selected_display = "ABC".index(row["selected_label"])
    expected_display = max(range(3), key=row["label_probabilities"].__getitem__)
    if selected_display != expected_display:
        raise ValueError("TriVUS selector selected label is not probability argmax")
    if row["selected_candidate_index"] != row["display_to_candidate"][selected_display]:
        raise ValueError("TriVUS selector selected-candidate mismatch")
    for key in ("benchmark", "setting", "row_id", "fold", "group", "image_sha256"):
        if row[key] != public[key]:
            raise ValueError(f"TriVUS selector public mismatch: {row['sample_key']}/{key}")
    if row["model_index_sha256"] != config["model"]["index_sha256"]:
        raise ValueError("TriVUS selector model hash mismatch")
    expected_prompt = hashlib.sha256(
        build_prompt(public, row["display_to_candidate"]).encode()
    ).hexdigest()
    if row["prompt_sha256"] != expected_prompt:
        raise ValueError("TriVUS selector prompt hash mismatch")
    if verify_overlay:
        overlay = render_overlay(
            public, row["display_to_candidate"], config["processor"]["max_rendered_edge"]
        )
        if row["overlay_sha256"] != rendered_image_sha256(overlay):
            raise ValueError("TriVUS selector overlay hash mismatch")
    elif len(row["overlay_sha256"]) != 64:
        raise ValueError("TriVUS selector overlay hash width mismatch")


def main():
    config = load_config()
    python = assert_selector_environment(config)
    if any((RUN_DIR / "data").glob("private*")):
        raise PermissionError("TriVUS blind lock requires no private labels")
    public_path = RUN_DIR / "data/public_records.jsonl"
    public_manifest = json.loads((RUN_DIR / "data/PUBLIC_MANIFEST.json").read_text())
    if sha256_file(public_path) != public_manifest["public_sha256"]:
        raise ValueError("TriVUS public hash drift")
    public_rows = load_jsonl(public_path)
    public = {row["sample_key"]: row for row in public_rows}
    predictions = []
    shard_report = {}
    for shard in range(config["inference"]["num_shards"]):
        path = RUN_DIR / f"selector/shards/shard-{shard}.jsonl"
        rows = load_jsonl(path)
        expected = sorted(public)[shard::config["inference"]["num_shards"]]
        if sorted(row["sample_key"] for row in rows) != expected:
            raise ValueError(f"TriVUS selector shard coverage mismatch: {shard}")
        predictions.extend(rows)
        shard_report[str(shard)] = {"path": str(path.relative_to(RUN_DIR)), "rows": len(rows), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if len(predictions) != config["expected_records"] or len({row["sample_key"] for row in predictions}) != len(predictions):
        raise ValueError("TriVUS selector merged coverage mismatch")
    for row in predictions:
        validate_prediction(row, public[row["sample_key"]], config)
    predictions.sort(key=lambda row: row["sample_key"])
    output = RUN_DIR / "selector/predictions.jsonl"
    manifest_path = RUN_DIR / "selector/BLIND_MANIFEST.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output)
    write_jsonl(output, predictions)
    manifest = {
        "schema_version": 1,
        "status": "PASS_TRIVUS_SELECTOR_BLIND_LOCK",
        "records": len(predictions),
        "settings": {setting: sum(row["setting"] == setting for row in predictions) for setting in ("low", "high")},
        "public_sha256": public_manifest["public_sha256"],
        "predictions_sha256": sha256_file(output),
        "model_index_sha256": config["model"]["index_sha256"],
        "python": python,
        "gpu_mapping": config["inference"]["gpu_mapping"],
        "shards": shard_report,
        "private_labels_created": False,
        "ground_truth_fields_used": False,
        "scorer_or_evaluator_imported": False,
        "label_metrics_computed": False,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()