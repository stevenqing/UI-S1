import argparse
import hashlib
import json
import subprocess
import sys
from functools import partial
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "repo/eval/android_control"
sys.path.insert(0, str(EVAL_DIR))

import android_control
from evaluate_android_control import evaluate_android_control_action
from qwen_vl_utils import smart_resize


EXPECTED_ROWS = 8444
MODEL_REVISION = "7b0e1de35afb807c6bfa70a2b85df24cf298e73d"
SOURCE_REVISION = "a4fca17809a4395ba1fe08d481bb82c790ea7236"
DATASET_REVISION = "92bf0d54e371474bff2a94dd93f087ec6940b54d"
EVALUATOR_SHA256 = "90ab6a56cae771245282c3ebcaf4cb96016d3c7c05072d155c7db91c5545e1f1"
EXPECTED_PROVENANCE = {
    "model_revision": MODEL_REVISION,
    "source_revision": SOURCE_REVISION,
    "dataset_revision": DATASET_REVISION,
    "backend": "vllm-0.11.0",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.65,
    "smart_resize_factor": 28,
    "temperature": 0.0,
    "max_tokens": 4096,
    "seed": 42,
}


def read_jsonl_snapshot(path: Path) -> tuple[list[dict], bytes]:
    snapshot = path.read_bytes()
    if snapshot and not snapshot.endswith(b"\n"):
        snapshot = snapshot.rsplit(b"\n", 1)[0] + b"\n"
    rows = []
    for line_number, line in enumerate(snapshot.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
    return rows, snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tool_call(output: str) -> dict:
    prediction = output.split("</think>")[-1]
    if "<tool_call>" in prediction:
        prediction = prediction.split("<tool_call>", 1)[1]
    else:
        prediction = (
            '{"name": "mobile_use", "arguments":'
            + prediction.split('{"name": "mobile_use", "arguments":', 1)[1]
        )
    if "</tool_call>" in prediction:
        prediction = prediction.split("</tool_call>", 1)[0]
    else:
        prediction = prediction.split("<conclusion>", 1)[0]
        prediction = prediction.rsplit("}}", 1)[0] + "}}"
    return json.loads(prediction.strip())["arguments"]


def format_prompt(messages: list[dict]) -> str:
    prompt = ""
    if messages[0]["role"] != "system":
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role in {"user", "human"}:
            content = content.replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
            )
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role in {"assistant", "gpt"}:
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return prompt + "<|im_start|>assistant\n"


def generate_reference_jobs(setting: str) -> list[dict]:
    android_control.smart_resize = partial(smart_resize, factor=28)
    evaluator = android_control.AndroidControl.__new__(android_control.AndroidControl)
    evaluator.eval_file = str(ROOT / "data/extracted/android_control_test.json")
    evaluator.eval_type = setting
    evaluator.max_pixels = 6400 * 28 * 28
    evaluator.system_prompt = (
        "You FIRST think about the reasoning process as an internal monologue and then "
        "provide the final answer.\nThe reasoning process MUST BE enclosed within "
        "<think> </think> tags.\nDuring the reasoning process, identify and state the "
        "sub-goal of the current step by enclosing it within <step> </step> tags."
    )
    evaluator.seed = 42
    evaluator.debug = False
    jobs = evaluator.generate_jobs()
    if len(jobs) != EXPECTED_ROWS:
        raise ValueError(f"reference job count is {len(jobs)}, expected {EXPECTED_ROWS}")
    if [job["question_id"] for job in jobs] != list(range(EXPECTED_ROWS)):
        raise ValueError("reference question IDs are not contiguous")
    return jobs


def record_mismatch(mismatches: dict[str, int], name: str) -> None:
    mismatches[name] = mismatches.get(name, 0) + 1


def audit_rows(rows: list[dict], jobs: list[dict], setting: str) -> dict:
    mismatches = {}
    parse_success = 0
    type_matches = 0
    exact_matches = 0
    click_matches = 0
    gt_clicks = 0

    question_ids = [row.get("question_id") for row in rows]
    if len(question_ids) != len(set(question_ids)):
        record_mismatch(mismatches, "duplicate_question_id")
    if question_ids != list(range(len(rows))):
        record_mismatch(mismatches, "question_id_order_or_coverage")

    identity_fields = (
        "question_id",
        "episode_id",
        "step_id",
        "image_path",
        "width",
        "height",
        "resized_width",
        "resized_height",
        "check_pams",
    )
    for index, row in enumerate(rows):
        if index >= len(jobs):
            record_mismatch(mismatches, "extra_row")
            continue
        job = jobs[index]
        for field in identity_fields:
            if row.get(field) != job[field]:
                record_mismatch(mismatches, f"field:{field}")

        prompt_sha256 = hashlib.sha256(format_prompt(job["messages"]).encode()).hexdigest()
        if row.get("prompt_sha256") != prompt_sha256:
            record_mismatch(mismatches, "prompt_sha256")
        if row.get("setting") != setting:
            record_mismatch(mismatches, "setting")
        for field, expected in EXPECTED_PROVENANCE.items():
            if row.get(field) != expected:
                record_mismatch(mismatches, f"provenance:{field}")

        prediction = None
        parse_error = None
        try:
            prediction = parse_tool_call(row["response"])
        except Exception as error:
            parse_error = f"{type(error).__name__}: {error}"
        if prediction != row.get("prediction"):
            record_mismatch(mismatches, "prediction")
        if row.get("parse_error") != parse_error:
            record_mismatch(mismatches, "parse_error")

        type_match = False
        exact_match = False
        if prediction is not None:
            parse_success += 1
            try:
                type_match, exact_match = evaluate_android_control_action(
                    prediction,
                    job["check_pams"],
                    job["width"],
                    job["height"],
                    job["resized_width"],
                    job["resized_height"],
                    pred_type="abs_resized",
                    gt_type="abs_resized",
                )
            except Exception:
                type_match = False
                exact_match = False
        type_match = bool(type_match)
        exact_match = bool(exact_match)
        if row.get("type_match") is not type_match:
            record_mismatch(mismatches, "type_match")
        if row.get("exact_match") is not exact_match:
            record_mismatch(mismatches, "exact_match")
        type_matches += type_match
        exact_matches += exact_match
        if exact_match and prediction is not None and prediction.get("action") == "click":
            click_matches += 1
        if job["check_pams"]["action"] == "click":
            gt_clicks += 1

    complete = len(rows) == EXPECTED_ROWS
    result = {
        "status": "PASS" if not mismatches else "FAIL",
        "coverage": "COMPLETE" if complete else "PARTIAL",
        "setting": setting,
        "rows": len(rows),
        "expected_rows": EXPECTED_ROWS,
        "unique_question_ids": len(set(question_ids)),
        "parse_success": parse_success,
        "mismatches": mismatches,
    }
    if complete and not mismatches:
        result["official_metrics_percent"] = {
            "type_accuracy": type_matches / EXPECTED_ROWS * 100,
            "grounding_accuracy": click_matches / gt_clicks * 100,
            "step_success_rate": exact_matches / EXPECTED_ROWS * 100,
        }
        result["official_counts"] = {
            "type_matches": type_matches,
            "grounding_matches": click_matches,
            "grounding_denominator": gt_clicks,
            "step_successes": exact_matches,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("high", "low"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    rows, snapshot = read_jsonl_snapshot(args.predictions)
    jobs = generate_reference_jobs(args.setting)
    result = audit_rows(rows, jobs, args.setting)
    result["artifact_sha256"] = hashlib.sha256(snapshot).hexdigest()
    result["evaluator_sha256"] = sha256_file(EVAL_DIR / "evaluate_android_control.py")
    result["source_head"] = subprocess.run(
        ["git", "-C", str(ROOT / "repo"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result["source_worktree"] = subprocess.run(
        ["git", "-C", str(ROOT / "repo"), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if result["source_head"] != SOURCE_REVISION:
        result["status"] = "FAIL"
        record_mismatch(result["mismatches"], "source_head")
    if result["evaluator_sha256"] != EVALUATOR_SHA256:
        result["status"] = "FAIL"
        record_mismatch(result["mismatches"], "evaluator_sha256")
    if result["source_worktree"]:
        result["status"] = "FAIL"
        record_mismatch(result["mismatches"], "source_worktree")
    if args.require_complete and result["coverage"] != "COMPLETE":
        result["status"] = "FAIL"
        record_mismatch(result["mismatches"], "incomplete_coverage")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()