import argparse
import ast
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as parquet
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


RUN_DIR = Path(__file__).resolve().parent
OFFICIAL_REPO = RUN_DIR / "repo"
OFFICIAL_REPO_REVISION = "3a397b078d6c14338f0646070212f8c3eb837881"
DATASET_REVISION = "6b4f69d9d3f947eb857301b6b01b2dba8e295b2c"
MODEL_REVISIONS = {
    "KDEGroup/UI-AGILE-3B": "84c28b06a7bda29a741139d64e227d176c0fb1c0",
    "KDEGroup/UI-AGILE": "de01366937b3c921f49ae1abe3b2c4a39b40ce8d",
    "LZXzju/Qwen2.5-VL-3B-UI-R1-E": "91c3e5f213ab3f42931e6398174f470c8500167f",
    "ritzzai/GUI-R1:GUI-R1-3B": "e74baccc4cfa77074e2d53e99a8244ab9fc2ca10",
    "ritzzai/GUI-R1:GUI-R1-7B": "e74baccc4cfa77074e2d53e99a8244ab9fc2ca10",
}
PROMPT_VARIABLES = {
    "android_control_detailed": "ANDROID_CONTROL_DETAILED",
    "ui_r1": "UI_R1_ANDROID_CONTROL",
    "gui_r1": "GUI_R1_ANDROID_CONTROL",
}
MODEL_TO_GT_ACTION_MAP = {
    "navigate_back": "press_back",
    "input_text": "type",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def load_prompt_template(name: str) -> str:
    source = OFFICIAL_REPO / "eval/android_control/inference_android_control.py"
    tree = ast.parse(source.read_text())
    variable = PROMPT_VARIABLES[name]
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                break
            return value
    raise ValueError(f"official prompt variable not found: {variable}")


def load_official_parsers():
    source_dir = OFFICIAL_REPO / "eval/android_control"
    sys.path.insert(0, str(source_dir))
    from eval import extract_param_value_loosely, gui_r1_extract_param
    from utils import extract_action, extract_coordinates

    return extract_action, extract_coordinates, extract_param_value_loosely, gui_r1_extract_param


def verify_official_repo() -> None:
    head = (OFFICIAL_REPO / ".git/HEAD").read_text().strip()
    if head != OFFICIAL_REPO_REVISION:
        raise ValueError(f"official repo revision mismatch: {head}")


def source_provenance(sample: dict) -> tuple[str, str]:
    image_bytes = sample["image"]["bytes"]
    image_sha256 = sha256_bytes(image_bytes)
    source_fields = {key: value for key, value in sample.items() if key != "image"}
    source_fields["image_sha256"] = image_sha256
    return image_sha256, canonical_sha256(source_fields)


def read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def build_request(sample: dict, processor, prompt_template: str) -> tuple[dict, dict]:
    image = Image.open(BytesIO(sample["image"]["bytes"])).convert("RGB")
    history = sample.get("history", "None")
    text_prompt = prompt_template.format(instruction=sample["instruction"], history=history)
    full_prompt = "<image>\n" + text_prompt
    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": full_prompt},
        ],
    }]
    prompt_for_model = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _, video_kwargs = process_vision_info(message, return_video_kwargs=True)
    processor_inputs = processor(
        text=[prompt_for_model], images=image_inputs, padding=True, return_tensors="pt"
    )
    image_grid = processor_inputs["image_grid_thw"][0]
    patch_size = processor.image_processor.patch_size
    resized_height = float(image_grid[1] * patch_size)
    resized_width = float(image_grid[2] * patch_size)
    origin_width, origin_height = image.size
    metadata = {
        "image_size": [origin_width, origin_height],
        "scale": [origin_width / resized_width, origin_height / resized_height],
        "text_prompt_sha256": sha256_bytes(text_prompt.encode()),
        "model_prompt_sha256": sha256_bytes(prompt_for_model.encode()),
    }
    request = {
        "prompt": prompt_for_model,
        "multi_modal_data": {"image": image_inputs},
        "mm_processor_kwargs": video_kwargs,
    }
    return request, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=MODEL_REVISIONS, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--data-setting", choices=("low", "high"), required=True)
    parser.add_argument("--prompt-template", choices=PROMPT_VARIABLES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    verify_official_repo()
    prompt_template = load_prompt_template(args.prompt_template)
    extract_action, extract_coordinates, extract_parameter, extract_gui_r1_parameter = (
        load_official_parsers()
    )

    table = parquet.read_table(args.data_path)
    if table.num_rows != 7708:
        raise ValueError(f"expected 7708 AndroidControl rows, found {table.num_rows}")
    rows = table.to_pylist()
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {args.output}")
    existing = read_existing(args.output) if args.resume else []
    completed = {row["index"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError("duplicate indices in existing output")
    if any(index % args.num_shards != args.shard_index for index in completed):
        raise ValueError("existing output contains an index from another shard")

    processor = AutoProcessor.from_pretrained(
        args.model_dir.resolve(), trust_remote_code=True, use_fast=False
    )
    model = LLM(
        model=str(args.model_dir.resolve()),
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.65,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        skip_special_tokens=False,
    )

    pending = [index for index in indices if index not in completed]
    with args.output.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.batch_size):
            batch_indices = pending[start : start + args.batch_size]
            requests = []
            request_metadata = []
            for index in batch_indices:
                request, metadata = build_request(rows[index], processor, prompt_template)
                requests.append(request)
                request_metadata.append(metadata)
            responses = model.generate(requests, sampling_params=sampling, use_tqdm=False)
            for index, metadata, response in zip(batch_indices, request_metadata, responses):
                sample = rows[index]
                generated_text = response.outputs[0].text
                predicted_action_raw = extract_action(generated_text)
                predicted_action = MODEL_TO_GT_ACTION_MAP.get(
                    predicted_action_raw, predicted_action_raw
                )
                predicted_coordinate, _, _ = extract_coordinates(generated_text)
                if predicted_coordinate is None:
                    predicted_coordinate = [0, 0]
                scaled_coordinate = [
                    predicted_coordinate[0] * metadata["scale"][0],
                    predicted_coordinate[1] * metadata["scale"][1],
                ]
                predicted_parameter = extract_parameter(generated_text)
                if args.prompt_template == "gui_r1":
                    predicted_parameter = extract_gui_r1_parameter(generated_text)
                image_sha256, source_sha256 = source_provenance(sample)
                ground_truth_input = sample["gt_input_text"]
                if sample["gt_action"] == "scroll":
                    ground_truth_input = ground_truth_input.lower()
                result = {
                    "index": index,
                    "data_setting": args.data_setting,
                    "instruction": sample["instruction"],
                    "history": sample.get("history"),
                    "gt_action": sample["gt_action"],
                    "gt_bbox": sample["gt_bbox"],
                    "gt_input_text": ground_truth_input,
                    "group": sample["group"],
                    "ui_type": sample["ui_type"],
                    "image_size": metadata["image_size"],
                    "resize_scale": metadata["scale"],
                    "pred_raw": generated_text,
                    "pred_action": predicted_action,
                    "pred_coord": scaled_coordinate,
                    "pred_input_text": predicted_parameter,
                    "image_sha256": image_sha256,
                    "source_sha256": source_sha256,
                    "text_prompt_sha256": metadata["text_prompt_sha256"],
                    "model_prompt_sha256": metadata["model_prompt_sha256"],
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
                    "shard_index": args.shard_index,
                }
                output.write(json.dumps(result, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()