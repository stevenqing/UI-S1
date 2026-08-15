import hashlib
import json
import os
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
OUTPUT_PATH = RUN_DIR / "ARM3_INVENTORY.json"
RAW_PATH = RUN_DIR / "raw/arm3_file_inventory.jsonl"
GRAN_MANIFEST = ROOT / "runs/gran/2026-08-14/INPUT_MANIFEST.json"
XFER_MANIFEST = ROOT / "runs/xfer/2026-08-07/PUBLICATION_MANIFEST.json"
VUS_BACKUP_MANIFEST = Path("/scratch/workspaceblobstore/visual-utility-selector/2026-08-11/BACKUP_MANIFEST.json")

GENERATION_KEYS = {
    "logprob", "logprobs", "token_logprob", "token_logprobs",
    "sequence_logprob", "sequence_score", "transition_scores",
}
TOKEN_KEYS = {"generated_token_ids", "output_token_ids", "token_ids"}
COORDINATE_SPAN_KEYS = {"coordinate_token_span", "coordinate_token_indices"}
SELECTOR_KEYS = {"label_logits", "label_probabilities", "selected_label"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recursive_keys(value, prefix=""):
    keys = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            keys.add(path)
            keys.update(recursive_keys(child, path))
    elif isinstance(value, list):
        for child in value[:1]:
            keys.update(recursive_keys(child, f"{prefix}[]"))
    return keys


def normalized_leaf_keys(keys):
    return {key.rsplit(".", 1)[-1].replace("[]", "").lower() for key in keys}


def inspect_jsonl(path, expected_sha, source_class, benchmark, source_path):
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"DECOMP Arm 3 trace mismatch: {source_path}")
    keys = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            keys.update(recursive_keys(value))
            rows += 1
    leaves = normalized_leaf_keys(keys)
    return {
        "benchmark": benchmark,
        "source_class": source_class,
        "source_path": source_path,
        "bytes": path.stat().st_size,
        "rows": rows,
        "sha256": expected_sha,
        "schema_keys": sorted(keys),
        "generation_logprobs_present": bool(leaves & GENERATION_KEYS),
        "token_ids_present": bool(leaves & TOKEN_KEYS),
        "coordinate_token_span_present": bool(leaves & COORDINATE_SPAN_KEYS),
        "selector_logits_present": bool(leaves & SELECTOR_KEYS),
    }


def screenspot_sources():
    manifest = json.loads(GRAN_MANIFEST.read_text())
    output = []
    allowed_roles = {
        "screenspot_gta1_views_0_15",
        "screenspot_qwen3_views_0_3",
        "screenspot_uitars_views_0_3",
    }
    for relative, info in manifest["files"].items():
        if not relative.endswith(".jsonl") or not (set(info["roles"]) & allowed_roles):
            continue
        output.append((ROOT / relative, info["sha256"], "generating_model_trace", "screenspot_pro", relative))
    if not output:
        raise ValueError("DECOMP Arm 3 found no ScreenSpot generating traces")
    return output


def mind2web_sources():
    manifest = json.loads(XFER_MANIFEST.read_text())
    output = []
    prefixes = ("raw/stage1/", "raw/stage2/", "raw/views/")
    for relative, info in manifest["artifacts"].items():
        if relative.endswith(".jsonl") and relative.startswith(prefixes):
            path = ROOT / "runs/xfer/2026-08-07" / relative
            output.append((path, info["sha256"], "generating_model_trace", "mind2web", f"runs/xfer/2026-08-07/{relative}"))
    if len(output) != 44:
        raise ValueError(f"DECOMP Arm 3 Mind2Web trace count mismatch: {len(output)}")
    return output


def selector_source():
    manifest = json.loads(VUS_BACKUP_MANIFEST.read_text())
    info = manifest["artifacts"]["zero_shot/predictions.jsonl"]
    return [(Path(info["backup_path"]), info["sha256"], "downstream_selector_control", "mixed", info["source_path"])]


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    if OUTPUT_PATH.exists() or RAW_PATH.exists():
        raise FileExistsError("DECOMP Arm 3 output exists")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if preflight["status"] != "PASS_DECOMP_PREFLIGHT_NO_ARM_STARTED" or preflight["labels_opened"] is not False:
        raise PermissionError("DECOMP Arm 3 preflight mismatch")
    records = [
        inspect_jsonl(*source)
        for source in screenspot_sources() + mind2web_sources() + selector_source()
    ]
    write_jsonl_fsynced(RAW_PATH, records)
    generating = [record for record in records if record["source_class"] == "generating_model_trace"]
    by_benchmark = {}
    for benchmark in ("screenspot_pro", "mind2web"):
        current = [record for record in generating if record["benchmark"] == benchmark]
        by_benchmark[benchmark] = {
            "files": len(current),
            "rows_across_files": sum(record["rows"] for record in current),
            "bytes": sum(record["bytes"] for record in current),
            "generation_logprob_files": sum(record["generation_logprobs_present"] for record in current),
            "token_id_files": sum(record["token_ids_present"] for record in current),
            "coordinate_token_span_files": sum(record["coordinate_token_span_present"] for record in current),
        }
    selector = [record for record in records if record["source_class"] == "downstream_selector_control"]
    output = {
        "schema_version": 1,
        "status": "DECOMP_ARM3_STOP_LOGPROB_CHANNEL_NOT_RETAINED",
        "labels_opened": False,
        "auroc_computed": False,
        "generating_model_logprob_available": False,
        "selector_logits_substituted": False,
        "benchmarks": by_benchmark,
        "selector_control": {
            "files": len(selector),
            "selector_logits_present": any(record["selector_logits_present"] for record in selector),
            "classification": "DOWNSTREAM_CANDIDATE_SELECTOR_NOT_GENERATING_MODEL_LOGPROB",
        },
        "forward_retention_policy": {
            "required": True,
            "fields": [
                "generated_token_ids", "decoded_response", "per_token_logprobs_or_unavailable_reason",
                "coordinate_token_span", "coordinate_logprob", "raw_sequence_score",
                "length_normalized_sequence_score", "normalization_formula", "sampling_parameters",
                "model_revision_and_index_hash", "prompt_and_image_hashes",
            ],
        },
        "raw": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "rows": len(records),
            "bytes": RAW_PATH.stat().st_size,
            "sha256": sha256_file(RAW_PATH),
            "write_flush_fsync_per_row": True,
        },
    }
    if any(record["generation_logprobs_present"] for record in generating):
        raise PermissionError("DECOMP Arm 3 unexpectedly found logprobs; committed amendment required")
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()