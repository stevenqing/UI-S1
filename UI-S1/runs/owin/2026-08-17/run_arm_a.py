import argparse
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MODEL_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/GTA1-7B"
MVP_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"
TRANSFORMERS_OVERLAY_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/mvp-overlay"
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
FORMAL_INPUT_PATH = RUN_DIR / "INFERENCE_INPUT_MANIFEST.jsonl"
SMOKE_INPUT_PATH = RUN_DIR / "SMOKE_INPUT_MANIFEST.jsonl"
AUTHORIZATION_PATH = RUN_DIR / "EXECUTION_AUTHORIZATION_006.json"
SMOKE_STATUS_PATH = RUN_DIR / "SMOKE_STATUS.json"
NONCE_MARKER_PATH = RUN_DIR / "raw/NONCE_CONSUMED.json"
FORMAL_SHARDS = ("common_11", "partial_1_10", "uncovered_0")
FORBIDDEN_INPUT_FIELDS = {"target_bbox", "bbox", "correct", "correctness", "reward", "label", "stratum", "target_center_contained", "target_bbox_contained"}
FORBIDDEN_TRACE_FIELDS = FORBIDDEN_INPUT_FIELDS | {"execution_shard"}
REQUIRED_TRACE_FIELDS = {"schema_version", "status", "sample_id", "row_id", "slot", "window", "image_sha256", "crop_sha256", "prompt_sha256", "model", "backend", "decoding", "decoded_response", "generated_token_ids", "per_token_logprobs", "logprobs_unavailable", "logprobs_unavailable_reason", "per_token_entropy", "per_token_top1_minus_top2_probability", "coordinate_token_span_indices", "aggregate_coordinate_logprob", "coordinate_token_entropy_mean", "coordinate_top1_minus_top2_probability_mean", "raw_sequence_logprob", "length_normalized_sequence_logprob", "sequence_normalization", "parsed"}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def recursive_keys(value):
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(recursive_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(recursive_keys(child))
        return result
    return set()


def validate_input_rows(rows, mode):
    expected_rows = 3 if mode == "smoke" else 500
    if len(rows) != expected_rows or len({row["row_id"] for row in rows}) != expected_rows:
        raise ValueError("OWIN input row mismatch")
    if recursive_keys(rows) & FORBIDDEN_INPUT_FIELDS:
        raise ValueError("OWIN inference input contains evaluation fields")
    for row in rows:
        if len(row["windows"]) != 12 or [window["slot"] for window in row["windows"]] != list(range(12)):
            raise ValueError(f"OWIN input window mismatch: {row['row_id']}")


def validate_trace_row(row):
    if not REQUIRED_TRACE_FIELDS <= set(row):
        raise ValueError(f"OWIN trace fields missing: {sorted(REQUIRED_TRACE_FIELDS - set(row))}")
    if recursive_keys(row) & FORBIDDEN_TRACE_FIELDS:
        raise ValueError("OWIN trace contains evaluation fields")


def parse_output(output_text, extract_coordinates):
    match = re.search(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", output_text)
    if match is None:
        return {"parse_status": "unparsable", "crop_local_point": None, "full_image_point": None}
    try:
        tuple(map(int, match.groups()))
        point = extract_coordinates(output_text)
    except Exception:
        return {"parse_status": "unparsable", "crop_local_point": None, "full_image_point": None}
    return {"parse_status": "parsed", "raw_model_point": list(point)}


def map_coordinate(raw_point, resized_size, working_size, resize, offset):
    pred_x, pred_y = raw_point
    if resize:
        pred_x //= 2
        pred_y //= 2
    local_x = int(pred_x * working_size[0] / resized_size[0])
    local_y = int(pred_y * working_size[1] / resized_size[1])
    return [local_x, local_y], [local_x + offset[0], local_y + offset[1]]


def coordinate_token_indices(processor, token_ids, output_text):
    match = re.search(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", output_text)
    if match is None:
        return []
    prefixes = [""]
    for index in range(1, len(token_ids) + 1):
        prefixes.append(processor.decode(token_ids[:index], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    if prefixes[-1] != output_text or any(len(prefixes[index]) < len(prefixes[index - 1]) for index in range(1, len(prefixes))):
        return []
    return [index for index in range(len(token_ids)) if len(prefixes[index]) < match.end() and len(prefixes[index + 1]) > match.start()]


def load_runtime(device):
    import torch

    sys.path[:0] = [str(MVP_ROOT), str(TRANSFORMERS_OVERLAY_ROOT)]
    import mvp_sspro

    config = mvp_sspro.Qwen2_5_VLConfig.from_pretrained(MODEL_ROOT)
    config.target_token_id = ","
    config.target_layer_idx = 20
    model = mvp_sspro.Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ROOT, config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device).eval()
    processor = mvp_sspro.Qwen2_5_VLProcessor.from_pretrained(MODEL_ROOT, min_pixels=3136, max_pixels=4096 * 2160)
    return torch, mvp_sspro, model, processor


def run_window(row, window, torch, mvp, model, processor, device, backend):
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    image_path = ROOT / row["image"]["path"]
    image = Image.open(image_path).convert("RGB")
    left, top, right, bottom = window["final_window"]
    selected = image if window["slot"] == 0 else image.crop((left, top, right, bottom))
    crop_buffer = io.BytesIO()
    selected.save(crop_buffer, format="PNG")
    resize = window["slot"] != 0
    if resize:
        selected = selected.resize((selected.width * 2, selected.height * 2))
    working_size = selected.size
    resized_height, resized_width = mvp.smart_resize(selected.height, selected.width, factor=processor.image_processor.patch_size * processor.image_processor.merge_size, min_pixels=processor.image_processor.min_pixels, max_pixels=processor.image_processor.max_pixels)
    resized = selected.resize((resized_width, resized_height))
    system_message = {"role": "system", "content": mvp.SYSTEM_PROMPT.format(height=resized_height, width=resized_width)}
    user_message = {"role": "user", "content": [{"type": "image", "image": resized}, {"type": "text", "text": row["instruction"]}]}
    image_inputs, video_inputs = process_vision_info([system_message, user_message])
    text = processor.apply_chat_template([system_message, user_message], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=32, do_sample=False, temperature=0.0, use_cache=True, pad_token_id=151645, return_dict_in_generate=True, output_scores=True)
    input_length = inputs.input_ids.shape[1]
    token_ids = generated.sequences[0, input_length:].detach().cpu().tolist()
    output_text = processor.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    per_logprob = []
    per_entropy = []
    per_margin = []
    unavailable_reason = None
    if len(generated.scores) != len(token_ids):
        unavailable_reason = f"score_count_{len(generated.scores)}_token_count_{len(token_ids)}"
    else:
        for token_id, score in zip(token_ids, generated.scores):
            logits = score[0].float()
            log_probabilities = torch.log_softmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            top_two = torch.topk(probabilities, 2).values
            per_logprob.append(float(log_probabilities[token_id].item()))
            per_entropy.append(float((-(probabilities * log_probabilities).sum()).item()))
            per_margin.append(float((top_two[0] - top_two[1]).item()))
    parsed = parse_output(output_text, mvp.extract_coordinates)
    coordinate_indices = coordinate_token_indices(processor, token_ids, output_text)
    if parsed["parse_status"] == "parsed":
        local, full = map_coordinate(parsed.pop("raw_model_point"), [resized_width, resized_height], list(working_size), resize, [left, top])
        parsed["crop_local_point"] = local
        parsed["full_image_point"] = full
    sequence_logprob = sum(per_logprob) if unavailable_reason is None else None
    trace = {
        "schema_version": 1,
        "status": "ok",
        "sample_id": row["sample_id"],
        "row_id": row["row_id"],
        "slot": window["slot"],
        "window": {key: value for key, value in window.items() if key not in {"target_center_contained", "target_bbox_contained"}},
        "image_sha256": row["image"]["sha256"],
        "crop_sha256": sha256_bytes(crop_buffer.getvalue()),
        "prompt_sha256": sha256_bytes(text.encode()),
        "model": {"id": "GTA1-7B", "revision": "701bedc80b447863bd60e3318ae44f6cbbfafd78", "index_sha256": sha256_file(MODEL_ROOT / "model.safetensors.index.json")},
        "backend": backend,
        "decoding": {"mode": "greedy", "do_sample": False, "temperature": 0.0, "top_p": None, "top_k": None, "seed": None, "max_new_tokens": 32, "pad_token_id": 151645, "use_cache": True},
        "decoded_response": output_text,
        "generated_token_ids": token_ids,
        "per_token_logprobs": per_logprob if unavailable_reason is None else None,
        "logprobs_unavailable": unavailable_reason is not None,
        "logprobs_unavailable_reason": unavailable_reason,
        "per_token_entropy": per_entropy if unavailable_reason is None else None,
        "per_token_top1_minus_top2_probability": per_margin if unavailable_reason is None else None,
        "coordinate_token_span_indices": coordinate_indices,
        "aggregate_coordinate_logprob": sum(per_logprob[index] for index in coordinate_indices) if unavailable_reason is None and coordinate_indices else None,
        "coordinate_token_entropy_mean": sum(per_entropy[index] for index in coordinate_indices) / len(coordinate_indices) if unavailable_reason is None and coordinate_indices else None,
        "coordinate_top1_minus_top2_probability_mean": sum(per_margin[index] for index in coordinate_indices) / len(coordinate_indices) if unavailable_reason is None and coordinate_indices else None,
        "raw_sequence_logprob": sequence_logprob,
        "length_normalized_sequence_logprob": sequence_logprob / len(token_ids) if sequence_logprob is not None and token_ids else None,
        "sequence_normalization": "sum_token_logprob_divided_by_generated_token_count",
        "parsed": parsed,
    }
    validate_trace_row(trace)
    return trace


def load_authorization():
    if not AUTHORIZATION_PATH.exists():
        raise PermissionError("OWIN GPU execution is not authorized")
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    if authorization.get("status") != "AUTHORIZED_ONE_TIME_OWIN_ARM_A_6000" or authorization.get("exact_formal_calls") != 6000:
        raise PermissionError("OWIN authorization mismatch")
    return authorization


def output_path(mode, shard):
    return RUN_DIR / "smoke/traces.jsonl" if mode == "smoke" else RUN_DIR / f"raw/arm_a_{shard}.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate-only", "smoke", "formal"), required=True)
    parser.add_argument("--shard", choices=FORMAL_SHARDS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    formal_rows = read_jsonl(FORMAL_INPUT_PATH)
    smoke_rows = read_jsonl(SMOKE_INPUT_PATH)
    validate_input_rows(formal_rows, "formal")
    validate_input_rows(smoke_rows, "smoke")
    if args.mode == "validate-only":
        print(json.dumps({"status": "PASS_OWIN_RUNNER_INPUT_VALIDATION", "formal_rows": len(formal_rows), "smoke_rows": len(smoke_rows), "gpu_used": False, "gpu_authorized": AUTHORIZATION_PATH.exists()}, indent=2))
        return
    authorization = load_authorization()
    if args.mode == "formal" and args.shard is None:
        raise ValueError("OWIN formal mode requires --shard")
    if args.mode == "formal":
        if not SMOKE_STATUS_PATH.exists() or json.loads(SMOKE_STATUS_PATH.read_text()).get("status") != "PASS_OWIN_SMOKE_36":
            raise PermissionError("OWIN formal run requires passing smoke")
        rows = [row for row in formal_rows if row["execution_shard"] == args.shard]
        expected = {"common_11": 200, "partial_1_10": 150, "uncovered_0": 150}[args.shard]
        if len(rows) != expected:
            raise ValueError("OWIN formal shard row mismatch")
        if not NONCE_MARKER_PATH.exists():
            write_json_atomic(NONCE_MARKER_PATH, {"status": "CONSUMED", "nonce_sha256": sha256_bytes(authorization["nonce"].encode()), "authorization_sha256": sha256_file(AUTHORIZATION_PATH)})
    else:
        rows = smoke_rows
    path = output_path(args.mode, args.shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    torch, mvp, model, processor = load_runtime(args.device)
    backend = {"name": "transformers_custom_qwen2_5_vl", "transformers_version": __import__("transformers").__version__, "torch_version": torch.__version__, "device": args.device}
    failures = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            for window in row["windows"]:
                try:
                    trace = run_window(row, window, torch, mvp, model, processor, args.device, backend)
                except Exception as error:
                    failures += 1
                    trace = {"schema_version": 1, "status": "failed", "sample_id": row["sample_id"], "row_id": row["row_id"], "slot": window["slot"], "window": window, "backend": backend, "error_type": type(error).__name__, "error": str(error)}
                handle.write(json.dumps(trace, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    total = len(rows) * 12
    summary = {"status": "PASS_OWIN_SMOKE_36" if args.mode == "smoke" and failures == 0 else "OWIN_SHARD_COMPLETE", "mode": args.mode, "shard": args.shard, "rows": len(rows), "calls": total, "failures": failures, "failure_rate": failures / total, "trace_path": str(path.relative_to(ROOT)), "trace_bytes": path.stat().st_size, "trace_sha256": sha256_file(path), "authorization_sha256": sha256_file(AUTHORIZATION_PATH)}
    status_path = SMOKE_STATUS_PATH if args.mode == "smoke" else RUN_DIR / f"raw/arm_a_{args.shard}_status.json"
    write_json_atomic(status_path, summary)
    print(json.dumps(summary, indent=2))
    if failures / total > 0.01 or (args.mode == "smoke" and failures):
        raise RuntimeError("OWIN failure threshold exceeded")


if __name__ == "__main__":
    main()