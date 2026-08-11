import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import smart_resize


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(VUS))

from vus_data import CANDIDATE_LABELS, COLORS, deterministic_permutation


FACTOR = 32
VUS_MIN_PIXELS = 200704
VUS_MAX_PIXELS = 1003520
RAVEL_MIN_PIXELS = 100352
RAVEL_MAX_PIXELS = 1003520
MAX_EDGE = 1600
FINE_FRACTION = 0.07
CONTEXT_FRACTION = 0.21
MODES = ("local", "random", "global_only", "fine_only", "context_only")


def _font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(path, size=size) if path.exists() else ImageFont.load_default()


def baseline_vus_budget(image_size):
    width, height = image_size
    scale = min(1.0, MAX_EDGE / max(width, height))
    width = max(1, round(width * scale))
    height = max(1, round(height * scale))
    resized_height, resized_width = smart_resize(
        height, width, factor=FACTOR,
        min_pixels=VUS_MIN_PIXELS, max_pixels=VUS_MAX_PIXELS,
    )
    return resized_width * resized_height, (resized_width, resized_height)


def dimensions_within_budget(image_size, target_pixels):
    width, height = image_size
    if target_pixels < RAVEL_MIN_PIXELS:
        raise ValueError(f"RAVEL target below minimum: {target_pixels}")
    if width * height <= target_pixels:
        scale = 2.0 * math.sqrt(target_pixels / (width * height))
        width = max(1, math.ceil(width * scale))
        height = max(1, math.ceil(height * scale))
    resized_height, resized_width = smart_resize(
        height, width, factor=FACTOR,
        min_pixels=RAVEL_MIN_PIXELS, max_pixels=int(target_pixels),
    )
    if resized_width * resized_height > target_pixels:
        raise ValueError("RAVEL smart resize exceeded target")
    if resized_width * resized_height < 0.85 * target_pixels:
        raise ValueError("RAVEL image underuses target pixel allocation")
    return resized_width, resized_height


def resize_unmarked(image, target_pixels):
    dimensions = dimensions_within_budget(image.size, target_pixels)
    return image.resize(dimensions, Image.Resampling.LANCZOS)


def _stable_random_center(sample_key, candidate_index, fraction, seed):
    payload = json.dumps(
        (sample_key, int(candidate_index), float(fraction), int(seed)),
        separators=(",", ":"),
    ).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    generator = __import__("numpy").random.default_rng(value)
    return float(generator.random()), float(generator.random())


def square_crop(image, coordinate, fraction, random_center=None):
    width, height = image.size
    side = max(8, round(min(width, height) * fraction))
    missing = coordinate is None
    out_of_frame = False
    if random_center is not None:
        normalized_x, normalized_y = random_center
    elif coordinate is None:
        normalized_x = normalized_y = 0.5
    else:
        normalized_x, normalized_y = map(float, coordinate)
        out_of_frame = not (0 <= normalized_x <= 1 and 0 <= normalized_y <= 1)
        normalized_x = min(1.0, max(0.0, normalized_x))
        normalized_y = min(1.0, max(0.0, normalized_y))
    center_x = round(normalized_x * (width - 1))
    center_y = round(normalized_y * (height - 1))
    left = center_x - side // 2
    top = center_y - side // 2
    right = left + side
    bottom = top + side
    canvas = Image.new("RGB", (side, side), "#222222")
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(width, right)
    source_bottom = min(height, bottom)
    if source_right > source_left and source_bottom > source_top:
        crop = image.crop((source_left, source_top, source_right, source_bottom))
        canvas.paste(crop, (source_left - left, source_top - top))
    return canvas, missing, out_of_frame


def render_mosaic(record, display_to_candidate, fraction, target_pixels, seed, random_centers=False):
    image = Image.open(record["image_path"]).convert("RGB")
    mosaic_width, mosaic_height = dimensions_within_budget((4, 3), target_pixels)
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height), "#111111")
    draw = ImageDraw.Draw(mosaic)
    for display_index, candidate_index in enumerate(display_to_candidate):
        column = display_index % 4
        row = display_index // 4
        left = round(column * mosaic_width / 4)
        right = round((column + 1) * mosaic_width / 4)
        top = round(row * mosaic_height / 3)
        bottom = round((row + 1) * mosaic_height / 3)
        cell_width = right - left
        cell_height = bottom - top
        tile_side = max(1, min(cell_width, cell_height))
        candidate = record["candidates"][candidate_index]
        random_center = (
            _stable_random_center(record["sample_key"], candidate_index, fraction, seed)
            if random_centers else None
        )
        crop, missing, out_of_frame = square_crop(
            image, candidate["coordinate"], fraction, random_center=random_center
        )
        crop = crop.resize((tile_side, tile_side), Image.Resampling.LANCZOS)
        paste_x = left + (cell_width - tile_side) // 2
        paste_y = top + (cell_height - tile_side) // 2
        mosaic.paste(crop, (paste_x, paste_y))
        center_x = paste_x + tile_side // 2
        center_y = paste_y + tile_side // 2
        color = COLORS[display_index]
        radius = max(3, tile_side // 18)
        draw.line((center_x - radius * 2, center_y, center_x + radius * 2, center_y), fill=color, width=max(2, radius // 2))
        draw.line((center_x, center_y - radius * 2, center_x, center_y + radius * 2), fill=color, width=max(2, radius // 2))
        draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline="white", width=max(1, radius // 3))
        font = _font(max(10, tile_side // 7))
        label = CANDIDATE_LABELS[display_index]
        draw.rectangle((left + 1, top + 1, left + max(18, tile_side // 4), top + max(16, tile_side // 5)), fill=color)
        draw.text((left + 4, top + 1), label, fill="white", font=font)
        if missing or out_of_frame:
            state = "NO POINT" if missing else "OUTSIDE"
            state_font = _font(max(8, tile_side // 12))
            draw.text((left + 3, bottom - max(14, tile_side // 8)), state, fill="#ffeb3b", font=state_font)
        draw.rectangle((left, top, right - 1, bottom - 1), outline="#666666", width=1)
    return mosaic


def candidate_lines(record, display_to_candidate):
    lines = []
    for display_index, candidate_index in enumerate(display_to_candidate):
        candidate = record["candidates"][candidate_index]
        coordinate = candidate["coordinate"]
        location = "no point" if coordinate is None else f"x={coordinate[0]:.4f}, y={coordinate[1]:.4f}"
        parameter = candidate.get("parameter", "")
        suffix = f", text={json.dumps(parameter, ensure_ascii=False)}" if parameter else ""
        lines.append(f"{CANDIDATE_LABELS[display_index]}: {candidate['action']} at {location}{suffix}")
    return lines


def evidence_prompt(record, display_to_candidate, mode):
    history = record.get("history") or []
    history_text = "\n".join(f"- {value}" for value in history) if history else "- none"
    if mode in ("local", "random"):
        image_description = (
            "Image 1 is the unmarked global interface. Image 2 shows fine candidate crops and "
            "Image 3 shows wider context candidate crops. Mosaic labels A-L match the candidate list."
        )
    elif mode == "global_only":
        image_description = "The image is the unmarked global interface. Use the listed normalized coordinates."
    elif mode == "fine_only":
        image_description = "The image is a fine candidate-crop mosaic. Mosaic labels A-L match the candidate list."
    elif mode == "context_only":
        image_description = "The image is a context candidate-crop mosaic. Mosaic labels A-L match the candidate list."
    else:
        raise ValueError(mode)
    return (
        "Select the best next GUI action. Reply with exactly one capital letter A through L.\n\n"
        f"{image_description}\n"
        f"Task: {record['instruction']}\n"
        f"Recent actions:\n{history_text}\n\n"
        "Candidates:\n" + "\n".join(candidate_lines(record, display_to_candidate)) + "\n"
    )


def render_evidence(record, mode, seed=20260811):
    if mode not in MODES:
        raise ValueError(mode)
    display_to_candidate = deterministic_permutation(record["sample_key"], 0, seed)
    image = Image.open(record["image_path"]).convert("RGB")
    baseline_pixels, baseline_dimensions = baseline_vus_budget(image.size)
    if mode in ("local", "random"):
        global_target = math.floor(0.50 * baseline_pixels)
        fine_target = math.floor(0.25 * baseline_pixels)
        context_target = baseline_pixels - global_target - fine_target
        images = [
            resize_unmarked(image, global_target),
            render_mosaic(record, display_to_candidate, FINE_FRACTION, fine_target, seed, random_centers=mode == "random"),
            render_mosaic(record, display_to_candidate, CONTEXT_FRACTION, context_target, seed, random_centers=mode == "random"),
        ]
        targets = [global_target, fine_target, context_target]
    elif mode == "global_only":
        images = [resize_unmarked(image, baseline_pixels)]
        targets = [baseline_pixels]
    elif mode == "fine_only":
        images = [render_mosaic(record, display_to_candidate, FINE_FRACTION, baseline_pixels, seed)]
        targets = [baseline_pixels]
    else:
        images = [render_mosaic(record, display_to_candidate, CONTEXT_FRACTION, baseline_pixels, seed)]
        targets = [baseline_pixels]
    image_pixels = [value.width * value.height for value in images]
    total_pixels = sum(image_pixels)
    ratio = total_pixels / baseline_pixels
    if ratio > 1.02 + 1e-12:
        raise ValueError(f"RAVEL-K1 expected pixel ratio: {ratio}")
    if ratio < 0.90 - 1e-12:
        raise ValueError(f"RAVEL token-match underuses budget: {ratio}")
    metadata = {
        "mode": mode,
        "baseline_vus_pixels": baseline_pixels,
        "baseline_vus_dimensions": list(baseline_dimensions),
        "target_pixels": targets,
        "image_dimensions": [[value.width, value.height] for value in images],
        "expected_processed_pixels": image_pixels,
        "expected_total_processed_pixels": total_pixels,
        "expected_pixel_ratio_vs_vus": ratio,
    }
    return images, display_to_candidate, evidence_prompt(record, display_to_candidate, mode), metadata
