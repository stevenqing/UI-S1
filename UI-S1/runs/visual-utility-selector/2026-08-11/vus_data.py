import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABELS = tuple("ABCDEFGHIJKLM")
CANDIDATE_LABELS = LABELS[:12]
KEEP_LABEL = LABELS[12]
FORBIDDEN_PUBLIC_KEYS = {
    "bbox", "candidate_success", "correct", "ground_truth", "gt", "label",
    "pos_candidates", "success", "target", "target_bbox",
}
COLORS = (
    "#d7191c", "#2c7bb6", "#008837", "#7b3294", "#e66101", "#1b7837",
    "#c51b7d", "#2166ac", "#b2182b", "#4d9221", "#762a83", "#a6611a",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*values):
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def deterministic_permutation(sample_key, epoch, seed):
    generator = np.random.default_rng(stable_seed(sample_key, int(epoch), int(seed)))
    return tuple(int(value) for value in generator.permutation(12))


def target_label(candidate_success, fallback_index, display_to_candidate, sample_key, epoch, seed):
    if len(candidate_success) != 12 or sorted(display_to_candidate) != list(range(12)):
        raise ValueError("VUS requires exactly 12 candidates and a complete permutation")
    if not 0 <= fallback_index < 12:
        raise ValueError("invalid fallback index")
    if candidate_success[fallback_index]:
        return KEEP_LABEL
    positives = [
        display_index
        for display_index, candidate_index in enumerate(display_to_candidate)
        if candidate_success[candidate_index]
    ]
    if not positives:
        return KEEP_LABEL
    chosen = stable_seed(sample_key, int(epoch), int(seed), "positive") % len(positives)
    return CANDIDATE_LABELS[positives[chosen]]


def candidate_to_display(display_to_candidate):
    inverse = [None] * 12
    for display_index, candidate_index in enumerate(display_to_candidate):
        inverse[candidate_index] = display_index
    if any(value is None for value in inverse):
        raise ValueError("incomplete candidate permutation")
    return tuple(inverse)


def _font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(path, size=size) if path.exists() else ImageFont.load_default()


def _marker_offsets(count, radius):
    if count == 1:
        return [(0.0, 0.0)]
    return [
        (
            math.cos(2 * math.pi * index / count) * radius * 2.4,
            math.sin(2 * math.pi * index / count) * radius * 2.4,
        )
        for index in range(count)
    ]


def render_overlay(record, display_to_candidate, max_edge=1600):
    image = Image.open(record["image_path"]).convert("RGB")
    original_width, original_height = image.size
    scale = min(1.0, max_edge / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            Image.Resampling.LANCZOS,
        )
    width, height = image.size
    draw = ImageDraw.Draw(image)
    radius = max(10, round(min(width, height) * 0.012))
    font = _font(max(12, round(radius * 1.15)))
    groups = {}
    for display_index, candidate_index in enumerate(display_to_candidate):
        coordinate = record["candidates"][candidate_index]["coordinate"]
        if coordinate is None:
            continue
        x = min(width - 1, max(0, round(float(coordinate[0]) * width)))
        y = min(height - 1, max(0, round(float(coordinate[1]) * height)))
        bucket = (round(x / max(1, radius)), round(y / max(1, radius)))
        groups.setdefault(bucket, []).append((display_index, x, y))
    for values in groups.values():
        offsets = _marker_offsets(len(values), radius)
        for (display_index, point_x, point_y), (offset_x, offset_y) in zip(values, offsets):
            center_x = min(width - radius - 1, max(radius + 1, round(point_x + offset_x)))
            center_y = min(height - radius - 1, max(radius + 1, round(point_y + offset_y)))
            color = COLORS[display_index]
            if center_x != point_x or center_y != point_y:
                draw.line((point_x, point_y, center_x, center_y), fill=color, width=max(2, radius // 4))
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=color, outline="white", width=max(2, radius // 5),
            )
            label = CANDIDATE_LABELS[display_index]
            box = draw.textbbox((0, 0), label, font=font)
            draw.text(
                (center_x - (box[2] - box[0]) / 2, center_y - (box[3] - box[1]) / 2 - box[1]),
                label, fill="white", font=font,
            )
    return image


def _candidate_line(display_index, candidate):
    coordinate = candidate["coordinate"]
    location = "no point" if coordinate is None else f"x={coordinate[0]:.4f}, y={coordinate[1]:.4f}"
    parameter = candidate.get("parameter", "")
    suffix = f", text={json.dumps(parameter, ensure_ascii=False)}" if parameter else ""
    return f"{CANDIDATE_LABELS[display_index]}: {candidate['action']} at {location}{suffix}"


def build_prompt(record, fallback_index, display_to_candidate):
    inverse = candidate_to_display(display_to_candidate)
    fallback_label = CANDIDATE_LABELS[inverse[fallback_index]]
    lines = [_candidate_line(index, record["candidates"][candidate_index]) for index, candidate_index in enumerate(display_to_candidate)]
    history = record.get("history") or []
    history_text = "\n".join(f"- {value}" for value in history) if history else "- none"
    return (
        "Select the best next GUI action from the labeled candidates in the screenshot. "
        "The current CEV choice is shown below. Choose M to keep it when no candidate is clearly better. "
        "Reply with exactly one capital letter A through M.\n\n"
        f"Task: {record['instruction']}\n"
        f"Recent actions:\n{history_text}\n\n"
        "Candidates:\n" + "\n".join(lines) + "\n"
        f"M: KEEP_CEV (same action as {fallback_label})\n"
    )


def build_candidate_prompt(record, display_to_candidate):
    lines = [_candidate_line(index, record["candidates"][candidate_index]) for index, candidate_index in enumerate(display_to_candidate)]
    history = record.get("history") or []
    history_text = "\n".join(f"- {value}" for value in history) if history else "- none"
    return (
        "Select the best next GUI action from the labeled candidates in the screenshot. "
        "Reply with exactly one capital letter A through L.\n\n"
        f"Task: {record['instruction']}\n"
        f"Recent actions:\n{history_text}\n\n"
        "Candidates:\n" + "\n".join(lines) + "\n"
    )


def audit_public_record(record):
    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = key.lower()
                if normalized in FORBIDDEN_PUBLIC_KEYS or normalized.startswith("gt_") or normalized.endswith("_bbox"):
                    raise ValueError(f"V-K1 forbidden public field: {path + [key]}")
                visit(child, path + [key])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [str(index)])
    visit(record, [])
    if len(record.get("candidates", [])) != 12:
        raise ValueError("VUS public record must contain 12 candidates")
    image_path = Path(record["image_path"])
    if not image_path.is_file() or sha256_file(image_path) != record["image_sha256"]:
        raise ValueError(f"VUS image hash mismatch: {record.get('sample_key')}")
    return True
