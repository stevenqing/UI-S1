import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MODEL_DIR = ROOT / "runs/mind2web-tongui/2026-07-28/models/Qwen2.5-VL-7B-Instruct"
DATA_DIR = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SOURCE_COMMIT = "2c1125067958df2468663004b2b4b7c50557da25"
SEED = 20260802


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_seed(row_id, stage):
    payload = f"{row_id}|{stage}|{SEED}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def smart_resize_target(height, width, min_pixels, max_pixels):
    current = height * width
    factor = 1.0
    if current < min_pixels:
        factor = math.sqrt(min_pixels / current)
    elif current > max_pixels:
        factor = math.sqrt(max_pixels / current)
    new_height = int(height * factor)
    new_width = int(width * factor)
    new_height = (new_height + 27) // 28 * 28
    new_width = (new_width + 27) // 28 * 28
    return new_width, new_height


def parse_box(text, width, height):
    try:
        match = re.search(r"\[(.*?)\]", text)
        if not match:
            return None
        values = [float(value) for value in match.group(1).replace(",", " ").split() if value.strip()]
        if len(values) != 4:
            return None
        x1, x2 = sorted((values[0], values[2]))
        y1, y2 = sorted((values[1], values[3]))
        return [
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        ]
    except (TypeError, ValueError):
        return None


def box_iou(left, right):
    if left is None or right is None:
        return 0.0
    lx1, lx2 = sorted((left[0], left[2]))
    ly1, ly2 = sorted((left[1], left[3]))
    rx1, rx2 = sorted((right[0], right[2]))
    ry1, ry2 = sorted((right[1], right[3]))
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(0, min(ly2, ry2) - max(ly1, ry1))
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 1e-6 else 0.0


def spatial_consistency(boxes):
    if len(boxes) <= 1:
        return 1.0
    values = [
        box_iou(left, right)
        for left_index, left in enumerate(boxes)
        for right_index, right in enumerate(boxes)
        if left_index != right_index
    ]
    return sum(values) / len(values) if values else 0.0


def adaptive_crop(candidates, width, height, sigma_scale=2.5, minimum=512):
    valid = []
    for candidate in candidates:
        box = candidate["box"]
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        valid.append({"center": [(x1 + x2) / 2, (y1 + y2) / 2], "size": [x2 - x1, y2 - y1]})
    if not valid:
        return None
    centers = np.asarray([value["center"] for value in valid])
    distances = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
    keep = max(1, int(len(valid) * 0.75))
    indices = np.argsort(distances)[:keep]
    filtered_centers = centers[indices]
    sizes = np.asarray([valid[index]["size"] for index in indices])
    sigma = np.sqrt(np.var(filtered_centers, axis=0) + np.mean(np.square(sizes / 4.0), axis=0))
    center = np.mean(filtered_centers, axis=0)
    side = max(2 * sigma_scale * sigma[0], 2 * sigma_scale * sigma[1], minimum)
    half = side / 2
    left, top, right, bottom = center[0] - half, center[1] - half, center[0] + half, center[1] + half
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    return [max(0, int(left)), max(0, int(top)), min(width, int(right)), min(height, int(bottom))]


class OfficialModel:
    def __init__(self):
        self.llm = LLM(
            model=str(MODEL_DIR), trust_remote_code=True, tensor_parallel_size=1,
            limit_mm_per_prompt={"image": 1}, max_model_len=16384,
            gpu_memory_utilization=0.65, enforce_eager=True,
            mm_processor_kwargs={"min_pixels": 10000, "max_pixels": 5000000},
        )

    def inference(self, instruction, image, count, temperature, seed):
        parameters = SamplingParams(
            n=count, temperature=temperature, top_p=1.0,
            max_tokens=128, stop=["<|endoftext|>", "<|im_end|>"],
            logprobs=1, seed=seed,
        )
        prompt = (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"Outline the position corresponding to the instruction: {instruction}. "
            "The output should be only [x1,y1,x2,y2].<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        outputs = self.llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            sampling_params=parameters, use_tqdm=False,
        )[0].outputs
        candidates = []
        for output in outputs:
            token_logprobs = []
            if output.logprobs:
                for index, values in enumerate(output.logprobs):
                    if index < len(output.token_ids) and output.token_ids[index] in values:
                        token_logprobs.append(values[output.token_ids[index]].logprob)
            confidence = math.exp(sum(token_logprobs) / len(token_logprobs)) if token_logprobs else 0.0
            text = output.text.strip()
            candidates.append({
                "response": text,
                "box": parse_box(text, image.width, image.height),
                "confidence": confidence,
            })
        return candidates


def select_reliable(candidates):
    votes = []
    for index, candidate in enumerate(candidates):
        votes.append(sum(
            box_iou(candidate["box"], other["box"]) > 0.5
            for other_index, other in enumerate(candidates)
            if other_index != index
        ))
    maximum = max(votes) if votes else 0
    if maximum > 0:
        indices = [index for index, value in enumerate(votes) if value == maximum]
        return max(indices, key=lambda index: candidates[index]["confidence"])
    return max(range(len(candidates)), key=lambda index: candidates[index]["confidence"])


def process_k(model, source, image, global_candidates, count):
    candidates = [value for value in global_candidates[:count] if value["box"] is not None]
    if not candidates:
        return {"point": [0.5, 0.5], "method": "fallback_center", "gate": None, "crop": None, "refinement": None, "forwards": count}
    consistency = spatial_consistency([value["box"] for value in candidates])
    confidence = sum(value["confidence"] for value in candidates) / len(candidates)
    score = consistency + confidence
    gate = {"spatial_consistency": consistency, "mean_confidence": confidence, "score": score, "reliable": score > 1.5}
    if gate["reliable"]:
        selected = candidates[select_reliable(candidates)]["box"]
        return {"point": [(selected[0] + selected[2]) / 2, (selected[1] + selected[3]) / 2], "method": "gating_pass", "gate": gate, "crop": None, "refinement": None, "forwards": count}
    crop = adaptive_crop(candidates, image.width, image.height)
    left, top, right, bottom = crop
    cropped = image.crop(crop)
    target_width, target_height = smart_resize_target(cropped.height, cropped.width, 1000000, 4000000)
    resized = cropped.resize((target_width, target_height), Image.Resampling.BICUBIC)
    refinement = model.inference(
        source["instruction"], resized, 1, 0.0,
        row_seed(source["id"], f"K{count}_refine"),
    )[0]
    if refinement["box"] is None:
        selected = max(candidates, key=lambda value: value["confidence"])["box"]
        point = [(selected[0] + selected[2]) / 2, (selected[1] + selected[3]) / 2]
        method = "crop_fail_fallback"
    else:
        box = refinement["box"]
        local_x = (box[0] + box[2]) / 2 * (right - left)
        local_y = (box[1] + box[3]) / 2 * (bottom - top)
        point = [(local_x + left) / image.width, (local_y + top) / image.height]
        method = "adaptive_crop_refine"
    return {"point": point, "method": method, "gate": gate, "crop": crop, "refinement": refinement, "forwards": count + 1}


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("F3 duplicate resumed identities")
    return set(ids)


def infer(args):
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any("bbox" in row or "target_bbox" in row for row in rows):
        raise ValueError("F3 requires complete label-free inputs")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    model = OfficialModel()
    model_hash = sha256_file(MODEL_DIR / "model.safetensors.index.json")
    written = 0
    with args.output.open("a", buffering=1) as output:
        for stable_index in indices:
            source = rows[stable_index]
            if source["id"] in completed:
                continue
            image = Image.open(DATA_DIR / "images" / source["img_filename"]).convert("RGB")
            target_width, target_height = smart_resize_target(image.height, image.width, 2000000, 4800000)
            resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
            baseline = model.inference(source["instruction"], resized, 1, 0.0, row_seed(source["id"], "baseline"))[0]
            baseline_point = None if baseline["box"] is None else [
                (baseline["box"][0] + baseline["box"][2]) / 2,
                (baseline["box"][1] + baseline["box"][3]) / 2,
            ]
            globals_ = model.inference(source["instruction"], resized, 8, 0.9, row_seed(source["id"], "global_K8"))
            k8 = process_k(model, source, image, globals_, 8)
            k3 = process_k(model, source, image, globals_, 3)
            predictions = {"baseline": {**baseline, "point": baseline_point}, "global_K8": globals_, "K8": k8, "K3": k3}
            artifact = {
                **source,
                "source_commit": SOURCE_COMMIT,
                "model_id": "Qwen2.5-VL-7B-Instruct",
                "model_revision": MODEL_REVISION,
                "model_index_sha256": model_hash,
                "global_resized_size": [target_width, target_height],
                "predictions": predictions,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            artifact["prediction_sha256"] = hashlib.sha256(
                json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            written += 1
            if written % 25 == 0:
                print(json.dumps({"shard": args.shard_index, "written": written}), flush=True)
    print(json.dumps({"status": "PASS", "shard": args.shard_index, "written": written}), flush=True)


def load_rows(paths):
    rows = {}
    expected_hash = sha256_file(MODEL_DIR / "model.safetensors.index.json")
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"F3 duplicate identity: {row['id']}")
            if row["shard_index"] != row["stable_index"] % 8 or row["num_shards"] != 8:
                raise ValueError(f"F3 shard mismatch: {row['id']}")
            if row["model_revision"] != MODEL_REVISION or row["model_index_sha256"] != expected_hash:
                raise ValueError(f"F3 model mismatch: {row['id']}")
            if row["source_commit"] != SOURCE_COMMIT:
                raise ValueError(f"F3 source mismatch: {row['id']}")
            if hashlib.sha256(json.dumps(row["predictions"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() != row["prediction_sha256"]:
                raise ValueError(f"F3 prediction hash mismatch: {row['id']}")
            if "bbox" in row or "target_bbox" in row:
                raise ValueError(f"F3 target leak: {row['id']}")
            if len(row["predictions"]["global_K8"]) != 8:
                raise ValueError(f"F3 K8 count mismatch: {row['id']}")
            rows[row["id"]] = row
    if len(rows) != 1581:
        raise ValueError(f"F3 requires 1,581 identities, found {len(rows)}")
    return rows


def score(args):
    rows = load_rows(args.shards)
    gta_rows = {}
    for shard in range(8):
        path = ROOT / f"runs/ccm-h2h/2026-07-31/h1/shards/top18/shard-{shard}.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                gta_rows[row["id"]] = row
    if set(rows) != set(gta_rows):
        raise ValueError("F3 trace/GT identity mismatch")
    outcomes = {method: {} for method in ("baseline", "K8", "K3")}
    method_counts = {method: {} for method in ("K8", "K3")}
    for row_id, row in rows.items():
        bbox = gta_rows[row_id]["target_bbox"]
        width, height = gta_rows[row_id]["img_size"]
        for method in outcomes:
            point = row["predictions"][method]["point"]
            hit = False if point is None else bbox[0] <= point[0] * width <= bbox[2] and bbox[1] <= point[1] * height <= bbox[3]
            outcomes[method][row_id] = hit
        for method in method_counts:
            name = row["predictions"][method]["method"]
            method_counts[method][name] = method_counts[method].get(name, 0) + 1
    accuracy = {method: sum(values.values()) / len(values) for method, values in outcomes.items()}
    anchor_pass = abs(accuracy["K8"] - 0.410) <= 0.01
    length_sensitive = accuracy["K8"] - accuracy["K3"] > 0.01
    if not anchor_pass:
        outcome = "anchor_fail"
    elif length_sensitive:
        outcome = "anchor_pass_microchain_length_sensitive"
    else:
        outcome = "anchor_pass_microchain_faithful"
    result = {
        "schema_version": 1,
        "status": "PASS",
        "rows": len(rows),
        "accuracy": accuracy,
        "reported": {"baseline": 0.276, "K8": 0.410},
        "delta_to_reported": {"baseline": accuracy["baseline"] - 0.276, "K8": accuracy["K8"] - 0.410},
        "anchor_pass": anchor_pass,
        "K8_minus_K3": accuracy["K8"] - accuracy["K3"],
        "microchain_length_sensitive": length_sensitive,
        "outcome": outcome,
        "method_counts": method_counts,
        "source": {"repository": "ZJU-REAL/UI-Zoomer", "commit": SOURCE_COMMIT},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inference = subparsers.add_parser("infer")
    inference.add_argument("--inputs", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--num-shards", type=int, default=8)
    inference.add_argument("--shard-index", type=int, required=True)
    inference.add_argument("--limit", type=int)
    inference.add_argument("--resume", action="store_true")
    scorer = subparsers.add_parser("score")
    scorer.add_argument("--shards", type=Path, nargs="+", required=True)
    scorer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"infer": infer, "score": score}[args.command](args)


if __name__ == "__main__":
    main()