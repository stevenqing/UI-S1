import hashlib
import json
import math
import os
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/selector_prereg.yaml"
RECOVERY_CONFIG_PATH = RUN_DIR / "configs/recovery.yaml"
LABELS = ("A", "B", "C")
COLORS = ("#d7191c", "#2c7bb6", "#008837")
ROW_FIELDS = {
    "schema_version", "sample_key", "benchmark", "setting", "row_id", "fold",
    "group", "image_path", "image_sha256", "instruction", "history", "candidates",
}
CANDIDATE_FIELDS = {"action", "coordinate", "parameter", "parse_ok"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", buffering=1) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_AFTER_R1_BEFORE_PUBLIC_BANK_AND_SELECTOR_INFERENCE":
        raise ValueError("TriVUS selector protocol is not frozen")
    if config.get("expected_records") != 4000 or config.get("candidates") != 3:
        raise ValueError("TriVUS selector coverage contract mismatch")
    if config.get("settings") != ["low", "high"]:
        raise ValueError("TriVUS selector setting order mismatch")
    if config.get("python") != ".venv-scaleup/bin/python":
        raise ValueError("TriVUS selector interpreter contract mismatch")
    if config.get("canonical_private_source_order") != [
        "UI-AGILE-7B", "GUI-R1-7B", "UI-R1-E-3B"
    ]:
        raise ValueError("TriVUS selector private source order mismatch")
    if config.get("public_candidate_order") != "sha256_sample_key_seed_public_bank_order":
        raise ValueError("TriVUS selector public candidate order mismatch")
    if (
        config.get("image_hash_semantics") != "extracted_png_file_bytes"
        or config.get("source_image_hash_semantics") != "original_parquet_compressed_bytes_provenance_only"
    ):
        raise ValueError("TriVUS selector image-hash semantics mismatch")
    if set(config.get("public_allowed_row_fields", ())) != ROW_FIELDS:
        raise ValueError("TriVUS selector row schema mismatch")
    if set(config.get("public_allowed_candidate_fields", ())) != CANDIDATE_FIELDS:
        raise ValueError("TriVUS selector candidate schema mismatch")
    normalization = config["candidate_normalization"]
    if normalization.get("parameter_max_chars") != 256 or normalization.get("history_max_chars") != 512:
        raise ValueError("TriVUS selector text limit mismatch")
    inference = config["inference"]
    if (
        inference.get("num_shards") != 8
        or inference.get("batch_size") != 1
        or inference.get("gpu_mapping") != {index: index for index in range(8)}
        or set(inference["gpu_mapping"].values()) != set(range(8))
    ):
        raise ValueError("TriVUS selector shard/GPU contract mismatch")
    dependencies = [config["recovery_manifest"], config["folds"]]
    for item in dependencies:
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"TriVUS selector dependency hash mismatch: {item['path']}")
    model = ROOT / config["model"]["path"]
    model_files = {
        "model.safetensors.index.json": "index_sha256",
        "config.json": "config_sha256",
        "preprocessor_config.json": "preprocessor_sha256",
        "tokenizer_config.json": "tokenizer_config_sha256",
    }
    for filename, key in model_files.items():
        if sha256_file(model / filename) != config["model"][key]:
            raise ValueError(f"TriVUS selector model hash mismatch: {filename}")
    return config


def assert_selector_environment(config):
    expected = (ROOT / config["python"]).absolute()
    observed = Path(sys.executable).absolute()
    if observed != expected:
        raise RuntimeError(f"TriVUS selector interpreter mismatch: {observed}/{expected}")
    return str(observed)


def audit_public_record(record, verify_image=True):
    if set(record) != ROW_FIELDS:
        raise ValueError(f"TriVUS public row schema mismatch: {set(record) ^ ROW_FIELDS}")
    if record["benchmark"] != "androidcontrol" or record["setting"] not in ("low", "high"):
        raise ValueError("TriVUS public benchmark/setting mismatch")
    if len(record["candidates"]) != 3:
        raise ValueError("TriVUS public row requires three candidates")
    for candidate in record["candidates"]:
        if set(candidate) != CANDIDATE_FIELDS:
            raise ValueError(f"TriVUS public candidate schema mismatch: {set(candidate) ^ CANDIDATE_FIELDS}")
        coordinate = candidate["coordinate"]
        if coordinate is not None:
            if (
                len(coordinate) != 2
                or not all(math.isfinite(float(value)) for value in coordinate)
                or not all(0.0 <= float(value) <= 1.0 for value in coordinate)
            ):
                raise ValueError("TriVUS public candidate coordinate mismatch")
        if not isinstance(candidate["parameter"], str) or len(candidate["parameter"]) > 256:
            raise ValueError("TriVUS public candidate parameter mismatch")
    if not isinstance(record["history"], str) or len(record["history"]) > 512:
        raise ValueError("TriVUS public history mismatch")
    if verify_image:
        image_path = ROOT / record["image_path"]
        if not image_path.is_file() or sha256_file(image_path) != record["image_sha256"]:
            raise ValueError(f"TriVUS public image mismatch: {record['sample_key']}")
    return True


def normalize_candidate(prediction, config):
    normalization = config["candidate_normalization"]
    action = str(prediction.get("action") or normalization["missing_action"])
    coordinate = prediction.get("position") if action in normalization["coordinate_actions"] else None
    if coordinate is not None:
        coordinate = [float(coordinate[0]), float(coordinate[1])]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinate):
            raise ValueError("TriVUS normalized coordinate outside [0,1]")
    parameter = (
        str(prediction.get("value") or "")[:normalization["parameter_max_chars"]]
        if action in normalization["parameter_actions"] else ""
    )
    return {
        "action": action,
        "coordinate": coordinate,
        "parameter": parameter,
        "parse_ok": bool(prediction.get("parse_ok")),
    }


def hash_permutation(sample_key, seed, suffix):
    digest = hashlib.sha256(f"{sample_key}/{seed}/{suffix}".encode()).digest()
    values = list(range(3))
    state = int.from_bytes(digest[:8], "big")
    for index in range(2, 0, -1):
        selected = state % (index + 1)
        values[index], values[selected] = values[selected], values[index]
        state //= index + 1
    return tuple(values)


def public_candidate_permutation(sample_key, seed):
    return hash_permutation(sample_key, seed, "public-bank-order")


def deterministic_permutation(sample_key, seed):
    return hash_permutation(sample_key, seed, "selector-display")


def _font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(path, size=size) if path.exists() else ImageFont.load_default()


def render_overlay(record, display_to_candidate, max_edge):
    with Image.open(ROOT / record["image_path"]) as source:
        image = source.convert("RGB")
    original_width, original_height = image.size
    scale = min(1.0, max_edge / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            Image.Resampling.LANCZOS,
        )
    width, height = image.size
    radius = max(10, round(min(width, height) * 0.015))
    font = _font(max(12, round(radius * 1.15)))
    draw = ImageDraw.Draw(image)
    for display_index, candidate_index in enumerate(display_to_candidate):
        coordinate = record["candidates"][candidate_index]["coordinate"]
        if coordinate is None:
            continue
        x = min(width - radius - 1, max(radius + 1, round(coordinate[0] * width)))
        y = min(height - radius - 1, max(radius + 1, round(coordinate[1] * height)))
        color = COLORS[display_index]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=max(2, radius // 5))
        label = LABELS[display_index]
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - box[1]), label, fill="white", font=font)
    return image


def rendered_image_sha256(image):
    digest = hashlib.sha256()
    digest.update(image.mode.encode())
    digest.update(json.dumps(image.size).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def candidate_line(display_index, candidate):
    coordinate = candidate["coordinate"]
    location = "no point" if coordinate is None else f"x={coordinate[0]:.4f}, y={coordinate[1]:.4f}"
    parameter = candidate["parameter"]
    suffix = f", text={json.dumps(parameter, ensure_ascii=False)}" if parameter else ""
    return f"{LABELS[display_index]}: {candidate['action']} at {location}{suffix}"


def build_prompt(record, display_to_candidate):
    lines = [
        candidate_line(display_index, record["candidates"][candidate_index])
        for display_index, candidate_index in enumerate(display_to_candidate)
    ]
    history = record["history"] or "None"
    return (
        "Select the best next GUI action from the labeled candidates. "
        "Reply with exactly one capital letter A, B, or C.\n\n"
        f"Task: {record['instruction']}\n"
        f"Recent history: {history}\n\n"
        "Candidates:\n" + "\n".join(lines) + "\n"
    )