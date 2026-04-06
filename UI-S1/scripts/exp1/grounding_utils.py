#!/usr/bin/env python3
"""
Shared utilities for grounding evaluation experiments.

Handles:
- Loading grounding samples from GUI-360 test set
- Smart resize for Qwen2.5-VL coordinate system
- Grounding prompt construction
- Coordinate parsing and transformation
- Multi-sample grounding calls with DBSCAN clustering
"""

import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp0.data_utils import DATASET_ROOT, is_coord_in_bbox, load_trajectory

# ---------------------------------------------------------------------------
# Grounding prompt (matches GUI-360 eval's GROUNDING_USER_PROMPT_QWEN)
# ---------------------------------------------------------------------------
GROUNDING_PROMPT = (
    "You are a helpful assistant. Given a screenshot of the current screen "
    "and user instruction, you need to output the position of the element "
    "you will operate.\n\n"
    "The instruction is:\n{instruction}\n\n"
    "Output the coordinate of the element you will operate within "
    "<coordinate></coordinate> tag:\n<coordinate> [x, y] </coordinate>"
)


# ---------------------------------------------------------------------------
# Smart resize (matches Qwen2.5-VL's image processor)
# ---------------------------------------------------------------------------
def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 56 * 56,
                 max_pixels: int = 14 * 14 * 4 * 1280) -> tuple[int, int]:
    """Adjust image dimensions to multiples of factor within pixel range."""
    new_height = int(np.round(height / factor) * factor)
    new_width = int(np.round(width / factor) * factor)
    new_pixels = new_height * new_width

    if new_pixels < min_pixels:
        scale = np.sqrt(min_pixels / new_pixels)
        new_height = int(np.round(new_height * scale / factor) * factor)
        new_width = int(np.round(new_width * scale / factor) * factor)
    elif new_pixels > max_pixels:
        scale = np.sqrt(max_pixels / new_pixels)
        new_height = int(np.round(new_height * scale / factor) * factor)
        new_width = int(np.round(new_width * scale / factor) * factor)

    return new_height, new_width


def preprocess_image(image_path: str) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """
    Load image, apply smart_resize, return base64 data URL + dimensions.

    Returns:
        (data_url, original_size_wh, resized_size_wh)
    """
    img = Image.open(image_path)
    orig_w, orig_h = img.size

    resized_h, resized_w = smart_resize(orig_h, orig_w)

    if (resized_w, resized_h) != (orig_w, orig_h):
        img = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    return data_url, (orig_w, orig_h), (resized_w, resized_h)


def transform_coord_to_original(coord: list[float],
                                original_wh: tuple[int, int],
                                resized_wh: tuple[int, int]) -> list[float]:
    """Transform coordinate from resized image space back to original space."""
    if original_wh == resized_wh:
        return coord
    scale_x = original_wh[0] / resized_wh[0]
    scale_y = original_wh[1] / resized_wh[1]
    return [coord[0] * scale_x, coord[1] * scale_y]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_coordinate_response(text: str) -> list[float] | None:
    """Parse coordinate from grounding response (<coordinate> tags, JSON, etc.)."""
    if not text:
        return None

    # 1. <coordinate> [x, y] </coordinate>
    m = re.search(r"<coordinate[^>]*>\s*\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?\s*</coordinate>",
                  text, re.IGNORECASE)
    if m:
        return [float(m.group(1)), float(m.group(2))]

    # 2. JSON "coordinates": [x, y]
    m = re.search(r'"coordinates"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
    if m:
        return [float(m.group(1)), float(m.group(2))]

    # 3. tool_call with coordinate arg
    try:
        tc_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if tc_match:
            action = json.loads(tc_match.group(1))
            args = action.get("args", {})
            if isinstance(args, str):
                args = json.loads(args)
            coord = args.get("coordinate", [])
            if isinstance(coord, list) and len(coord) == 2:
                return [float(coord[0]), float(coord[1])]
    except Exception:
        pass

    # 4. Bare [x, y]
    m = re.search(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
    if m:
        return [float(m.group(1)), float(m.group(2))]

    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_grounding_samples(dataset_root: str = None, max_samples: int = 0) -> list[dict]:
    """Load grounding samples from GUI-360 test set directory structure."""
    if dataset_root is None:
        dataset_root = str(DATASET_ROOT)
    test_root = Path(dataset_root) / "test"
    data_path = test_root / "data"
    image_path = test_root / "image"

    samples = []
    for domain in sorted(data_path.iterdir()):
        if not domain.is_dir():
            continue
        for category in sorted(domain.iterdir()):
            if not category.is_dir():
                continue
            success_dir = category / "success"
            if not success_dir.exists():
                continue

            for jsonl_file in sorted(success_dir.glob("*.jsonl")):
                steps = load_trajectory(jsonl_file)
                for i, step_data in enumerate(steps):
                    step = step_data.get("step", {})
                    action = step.get("action", {})
                    tags = step.get("tags", [])

                    if "grounding" not in tags:
                        continue
                    if not action.get("rectangle"):
                        continue

                    screenshot = image_path / domain.name / category.name / step.get("screenshot_clean", "")
                    if not screenshot.exists():
                        continue

                    sample = {
                        "sample_id": f"{domain.name}_{category.name}_{jsonl_file.stem}_{i+1}",
                        "domain": domain.name,
                        "category": category.name,
                        "thought": step.get("thought", ""),
                        "screenshot": str(screenshot),
                        "gt_function": action.get("function", ""),
                        "gt_args": action.get("args", {}),
                        "gt_rectangle": action.get("rectangle", {}),
                    }
                    samples.append(sample)

                    if max_samples > 0 and len(samples) >= max_samples:
                        return samples

    return samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_grounding(pred_coord: list[float] | None,
                       gt_rectangle: dict) -> dict:
    """Evaluate predicted coordinate against ground truth bbox."""
    if pred_coord is None:
        return {"correct": False, "distance": float("inf")}

    left = gt_rectangle.get("left", 0)
    top = gt_rectangle.get("top", 0)
    right = gt_rectangle.get("right", 0)
    bottom = gt_rectangle.get("bottom", 0)

    x, y = pred_coord[0], pred_coord[1]
    correct = (left <= x <= right) and (top <= y <= bottom)

    # Distance to bbox center
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    return {"correct": correct, "distance": float(distance)}


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------
def call_grounding_once(client, model_name: str, sample: dict,
                        temperature: float = 0.0,
                        max_tokens: int = 512) -> tuple[list[float] | None, str]:
    """
    Call model once with grounding prompt.
    Returns (transformed_coord, raw_response).
    """
    data_url, orig_wh, resized_wh = preprocess_image(sample["screenshot"])
    prompt = GROUNDING_PROMPT.format(instruction=sample["thought"])

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]
    }]

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  API call failed: {e}")
        return None, ""

    coord = parse_coordinate_response(text)
    if coord is not None:
        coord = transform_coord_to_original(coord, orig_wh, resized_wh)

    return coord, text


def call_grounding_k_times(client, model_name: str, sample: dict,
                           K: int, temperature: float = 0.7,
                           max_tokens: int = 512) -> list[tuple[list[float] | None, str]]:
    """
    Call model K times with grounding prompt at temperature > 0.
    Uses n=K parameter for batched inference (single API call → K completions).
    Returns list of (transformed_coord, raw_response).
    """
    data_url, orig_wh, resized_wh = preprocess_image(sample["screenshot"])
    prompt = GROUNDING_PROMPT.format(instruction=sample["thought"])

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]
    }]

    results = []
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=K,
        )
        for choice in resp.choices:
            text = choice.message.content or ""
            coord = parse_coordinate_response(text)
            if coord is not None:
                coord = transform_coord_to_original(coord, orig_wh, resized_wh)
            results.append((coord, text))
    except Exception as e:
        # Fallback: n parameter not supported, call K times sequentially
        print(f"  Batched K-sample failed ({e}), falling back to sequential")
        for _ in range(K):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
            except Exception as e2:
                print(f"  K-sample API call failed: {e2}")
                text = ""

            coord = parse_coordinate_response(text)
            if coord is not None:
                coord = transform_coord_to_original(coord, orig_wh, resized_wh)
            results.append((coord, text))

    return results


# ---------------------------------------------------------------------------
# DBSCAN clustering (re-exported from exp0 for convenience)
# ---------------------------------------------------------------------------
def cluster_coordinates(coords: list[list], eps: float = 30.0, min_samples: int = 1) -> dict:
    """
    Cluster K coordinate predictions using DBSCAN.
    Returns cluster_center, agreement_rate, num_clusters, etc.
    """
    from sklearn.cluster import DBSCAN

    if not coords or len(coords) == 0:
        return {
            "cluster_center": None,
            "agreement_rate": 0.0,
            "num_clusters": 0,
            "is_multimodal": False,
            "coord_std": (float("inf"), float("inf")),
        }

    if len(coords) == 1:
        return {
            "cluster_center": coords[0],
            "agreement_rate": 1.0,
            "num_clusters": 1,
            "is_multimodal": False,
            "coord_std": (0.0, 0.0),
        }

    X = np.array(coords)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    unique_labels = set(labels) - {-1}
    num_clusters = len(unique_labels)

    if num_clusters == 0:
        # All noise
        center = np.mean(X, axis=0).tolist()
        return {
            "cluster_center": center,
            "agreement_rate": 0.0,
            "num_clusters": 0,
            "is_multimodal": False,
            "coord_std": (float(np.std(X[:, 0])), float(np.std(X[:, 1]))),
        }

    # Find largest cluster
    cluster_sizes = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
    largest_label = max(cluster_sizes, key=cluster_sizes.get)
    largest_mask = labels == largest_label
    largest_points = X[largest_mask]

    center = np.mean(largest_points, axis=0).tolist()
    agreement_rate = float(cluster_sizes[largest_label] / len(coords))

    return {
        "cluster_center": center,
        "agreement_rate": agreement_rate,
        "num_clusters": num_clusters,
        "is_multimodal": num_clusters > 1,
        "coord_std": (float(np.std(X[:, 0])), float(np.std(X[:, 1]))),
    }
