import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

from PIL import Image
import yaml
from qwen_vl_utils import smart_resize
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
CONFIG_PATH = RUN_DIR / "configs/g1_roster.yaml"
INPUT_SHA256 = "0e6b4387f704b94ec071c8fdb6a381c3293f2bfe8b9ae846b613529c476061b8"
POINT_PATTERN = re.compile(r"(?:<point>\s*)?[\[(]?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]?", re.I)
BBOX_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("G1 duplicate resumed identities")
    return set(ids)


def parse_response(response, model_name, original_size, resized_size):
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    if model_name == "UI-Venus-Ground-72B":
        match = BBOX_PATTERN.search(response)
        if not match:
            return [0.0, 0.0], False
        left, top, right, bottom = map(float, match.groups())
        values = (left, top, right, bottom)
        if not all(math.isfinite(value) for value in values):
            return [0.0, 0.0], False
        return [
            (left + right) / 2 * original_width / resized_width,
            (top + bottom) / 2 * original_height / resized_height,
        ], True
    match = POINT_PATTERN.search(response)
    if not match:
        return [0.0, 0.0], False
    x, y = map(float, match.groups())
    if not math.isfinite(x) or not math.isfinite(y):
        return [0.0, 0.0], False
    if model_name == "GTA1-72B":
        if not 0 <= x <= resized_width or not 0 <= y <= resized_height:
            return [0.0, 0.0], False
        return [x * original_width / resized_width, y * original_height / resized_height], True
    if not 0 <= x <= 1000 or not 0 <= y <= 1000:
        return [0.0, 0.0], False
    return [x / 1000 * original_width, y / 1000 * original_height], True


def prompt_for(model_name, spec, processor, image, instruction):
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=factor,
        min_pixels=spec["min_pixels"],
        max_pixels=spec["max_pixels"],
    )
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    if model_name == "UI-Venus-Ground-72B":
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": spec["prompt"].format(instruction=instruction)},
        ]}]
    else:
        system_prompt = spec["system_prompt"].format(height=resized_height, width=resized_width)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
        ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_name == "Qwen3.5-122B-A10B":
        kwargs["enable_thinking"] = False
    prompt = processor.apply_chat_template(messages, **kwargs)
    return prompt, resized, [resized_width, resized_height]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", choices=("GTA1-72B", "UI-Venus-Ground-72B", "Qwen3.5-122B-A10B"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.68)
    args = parser.parse_args()
    if sha256_file(args.inputs) != INPUT_SHA256:
        raise ValueError("G1 label-free input hash mismatch")
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any("bbox" in row or "target_bbox" in row for row in rows):
        raise ValueError("G1 requires complete label-free inputs")
    if args.limit is not None:
        rows = rows[:args.limit]
    config = yaml.safe_load(CONFIG_PATH.read_text())
    spec = config["models"][args.model]
    if args.model_dir.name != args.model or not (args.model_dir / "model.safetensors.index.json").is_file():
        raise ValueError("G1 model directory mismatch")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model_index_hash = sha256_file(args.model_dir / "model.safetensors.index.json")
    protocol_hash = canonical_hash(spec)
    max_model_len = 16384
    engine = LLM(
        model=str(args.model_dir),
        tensor_parallel_size=8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": spec["min_pixels"], "max_pixels": spec["max_pixels"]},
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=spec["max_new_tokens"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    pending = [row for row in rows if row["id"] not in completed]
    written = 0
    with args.output.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            requests = []
            metadata = []
            for source in batch:
                image = Image.open(DATA_ROOT / "images" / source["img_filename"]).convert("RGB")
                prompt, resized, resized_size = prompt_for(args.model, spec, processor, image, source["instruction"])
                requests.append({"prompt": prompt, "multi_modal_data": {"image": resized}})
                metadata.append((source, resized_size))
            generated = engine.generate(requests, sampling, use_tqdm=False)
            if len(generated) != len(metadata):
                raise ValueError("G1 generation batch size mismatch")
            for result, (source, resized_size) in zip(generated, metadata):
                response = result.outputs[0].text
                point, parse_ok = parse_response(response, args.model, source["img_size"], resized_size)
                prediction = {
                    "point": point,
                    "parse_ok": parse_ok,
                    "response": response,
                    "resized_size": resized_size,
                }
                artifact = {
                    **source,
                    "model_id": args.model,
                    "model_revision": spec["revision"],
                    "model_index_sha256": model_index_hash,
                    "protocol_sha256": protocol_hash,
                    "prediction": prediction,
                }
                artifact["prediction_sha256"] = canonical_hash(prediction)
                output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
                written += 1
            print(json.dumps({"model": args.model, "written": written, "total_pending": len(pending)}), flush=True)
    print(json.dumps({"status": "PASS", "model": args.model, "written": written}), flush=True)


if __name__ == "__main__":
    main()
