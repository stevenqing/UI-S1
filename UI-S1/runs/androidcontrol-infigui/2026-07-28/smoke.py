import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "repo/eval/android_control"
sys.path.insert(0, str(EVAL_DIR))

import android_control
from qwen_vl_utils import smart_resize
from evaluate_android_control import evaluate_android_control_action


android_control.Qwen2VL = partial(
    android_control.Qwen2VL,
    gpu_memory_utilization=0.65,
)
android_control.smart_resize = partial(smart_resize, factor=28)
AndroidControl = android_control.AndroidControl


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("high", "low"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    evaluator = AndroidControl(
        model_path=str(ROOT / "model/InfiGUI-R1-3B"),
        eval_file=str(ROOT / "data/extracted/android_control_test.json"),
        image_root=str(ROOT / "data/extracted"),
        output_dir=str(args.output_dir),
        eval_type=args.setting,
        thinking=True,
        tensor_parallel_size=2,
        max_num_seqs=2,
        seed=42,
        num_processes=1,
    )
    jobs = evaluator.generate_jobs()[: args.limit]
    jobs = evaluator.inference(jobs, temperature=0.0, max_tokens=4096, seed=42)

    for job in jobs:
        prediction = parse_tool_call(job["llm_output"])
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
        job["llm_prediction"] = prediction
        job["type_match"] = type_match
        job["exact_match"] = exact_match

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "status": "PASS",
        "setting": args.setting,
        "jobs": len(jobs),
        "type_matches": sum(job["type_match"] for job in jobs),
        "exact_matches": sum(job["exact_match"] for job in jobs),
        "model_revision": "7b0e1de35afb807c6bfa70a2b85df24cf298e73d",
        "source_revision": "a4fca17809a4395ba1fe08d481bb82c790ea7236",
        "dataset_revision": "92bf0d54e371474bff2a94dd93f087ec6940b54d",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.65,
        "smart_resize_factor": 28,
        "temperature": 0.0,
        "max_tokens": 4096,
        "seed": 42,
        "jobs_sha256": hashlib.sha256(
            json.dumps(jobs, sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest(),
    }
    (args.output_dir / "smoke_summary.json").write_text(
        json.dumps(artifact, indent=2) + "\n"
    )
    (args.output_dir / "jobs.json").write_text(
        json.dumps(jobs, indent=2, ensure_ascii=True) + "\n"
    )
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()