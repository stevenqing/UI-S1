"""
OS-Genesis 7B Evaluation with History Enabled

This version enables action history to test if it improves multi-step performance.
Results are logged to OS_Genesis_7b_with_history.jsonl for comparison.
"""

import argparse
import copy
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from agentcpm_utils import map_action_space2qwenvl
from os_genesis_utils import (build_history_actions_str, os_gensis_2minicpm,
                            predict)
from PIL import Image

from qwenvl_utils import evaluate_android_control_action
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from x.data.agent.json import JsonFormat
from x.qwen.data_format import slim_messages

result_lock = Lock()

def fix_line(line):
    for step in line['steps']:
        check_options = copy.deepcopy(step['action_content'])
        if 'candidate_bbox' in step:
            continue
        if 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        else:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options
    return line


def format_action_for_history(action_minicpm, model_response):
    """
    Format the parsed action into a readable history string.
    """
    # Extract thought if present in model response
    thought = ""
    if "thought:" in model_response.lower():
        thought_start = model_response.lower().find("thought:")
        thought_end = model_response.lower().find("action:")
        if thought_end == -1:
            thought_end = len(model_response)
        thought = model_response[thought_start + len("thought:"):thought_end].strip()[:200]

    # Format action based on type
    if "POINT" in action_minicpm:
        point = action_minicpm["POINT"]
        if "to" in action_minicpm:
            action_desc = f"SCROLL {action_minicpm['to'].upper()} at ({point[0]}, {point[1]})"
        elif "duration" in action_minicpm and action_minicpm.get("duration", 0) >= 1000:
            action_desc = f"LONG_PRESS at ({point[0]}, {point[1]})"
        else:
            action_desc = f"CLICK at ({point[0]}, {point[1]})"
    elif "TYPE" in action_minicpm:
        action_desc = f"TYPE \"{action_minicpm['TYPE'][:50]}\""
    elif "PRESS" in action_minicpm:
        action_desc = f"PRESS_{action_minicpm['PRESS']}"
    elif "OPEN_APP" in action_minicpm:
        action_desc = f"OPEN_APP \"{action_minicpm['OPEN_APP']}\""
    else:
        action_desc = str(action_minicpm)

    if thought:
        return f"{thought} -> {action_desc}"
    return action_desc


def process_line(line, args):
    num_steps = len(line['steps'])
    state = None
    model_response = None
    step_id = 0
    task_success = False
    fixed_line = fix_line(line)
    history_list = []
    step_details = []  # Track details for each step

    try:
        while step_id < num_steps:
            step = fixed_line['steps'][step_id]
            image_path = step['screenshot']
            low_instruction = step['step_instruction'] if 'step_instruction' in step else ''
            current_check_pam = step['check_options']

            # ENABLED: Build history from previous actions
            history = build_history_actions_str(history_list)

            image = Image.open(image_path)
            width, height = image.size

            # Call model prediction with history
            model_response = predict(
                model_name=args.model_name,
                instruction=line['goal'],
                low_instruction=low_instruction,  # Now passing low_instruction
                history=history,
                image=image
            )

            print(f"=== Step {step_id + 1}/{num_steps} ===")
            print(f"History length: {len(history_list)} steps")
            print(f"Model Response: {model_response[:500] if model_response else 'None'}")
            print("=== End Response ===")

            action_minicpm = os_gensis_2minicpm(model_response)
            print("Action Minicpm:", action_minicpm)

            # Build history entry from actual model action (not just low_instruction)
            if model_response:
                history_entry = format_action_for_history(action_minicpm, model_response)
            else:
                history_entry = low_instruction if low_instruction else f"Step {step_id + 1}"
            history_list.append(history_entry)

            # OS-Genesis outputs absolute pixel coordinates
            pred_action = map_action_space2qwenvl(action_minicpm, [width, height], coordinate_format="absolute")
            print("Predicted Action:", pred_action)

            type_match, extract_match = evaluate_android_control_action(
                pred_action,
                current_check_pam,
                width, height,
                width, height,
                ignore_actions=[]
            )

            # Track step details for logging
            step_details.append({
                "step_id": step_id,
                "type_match": type_match,
                "extract_match": extract_match,
                "history_used": len(history_list) - 1,
                "action": str(action_minicpm)[:200]
            })

            print("Type Match:", type_match)
            print("Extract Match:", extract_match)

            if not extract_match:
                break

            step_id += 1

        task_success = (step_id == num_steps)

    except Exception as e:
        print(f"Error processing goal '{line['goal']}': {e}")
        traceback.print_exc()
        task_success = False
        step_id = 0

    # Build result with more detailed logging
    result = {
        "goal": line['goal'],
        "num_steps": num_steps,
        "task_success": task_success,
        "final_step_id": step_id,
        "history_enabled": True,
        "step_details": step_details,
        "final_history_length": len(history_list)
    }

    # Write to separate file for history-enabled version
    with result_lock:
        result_path = os.path.join(args.output_dir, f"OS_Genesis_7b_with_history.jsonl")
        with open(result_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return result


def main(args):
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Read data
    std_data = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            std_data.append(json.loads(line))

    print(f"=== OS-Genesis Evaluation WITH HISTORY ===")
    print(f"Loaded {len(std_data)} tasks. Starting parallel evaluation...")
    print(f"Results will be saved to: {args.output_dir}/OS_Genesis_7b_with_history.jsonl")

    # Parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_line = {executor.submit(process_line, line, args): line for line in std_data}
        for future in as_completed(future_to_line):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Task generated an exception: {e}")

    # Calculate statistics
    success_count = sum(1 for r in results if r["task_success"])
    success_rate = success_count / len(results) * 100 if results else 0
    avg_progress = sum(r["final_step_id"] / r['num_steps'] for r in results) / len(results) if results else 0.0

    # Calculate single-step vs multi-step stats
    single_step_results = [r for r in results if r["num_steps"] == 1]
    multi_step_results = [r for r in results if r["num_steps"] > 1]

    single_step_success = sum(1 for r in single_step_results if r["task_success"])
    multi_step_success = sum(1 for r in multi_step_results if r["task_success"])

    single_step_rate = (single_step_success / len(single_step_results) * 100) if single_step_results else 0
    multi_step_rate = (multi_step_success / len(multi_step_results) * 100) if multi_step_results else 0

    print(f"\n=== Evaluation Results (WITH HISTORY) ===")
    print(f"Overall Success Rate: {success_rate:.2f}% ({success_count}/{len(results)})")
    print(f"Single-Step Success Rate: {single_step_rate:.2f}% ({single_step_success}/{len(single_step_results)})")
    print(f"Multi-Step Success Rate: {multi_step_rate:.2f}% ({multi_step_success}/{len(multi_step_results)})")
    print(f"Average Progress: {avg_progress:.2f}")

    # Save summary
    summary = {
        "experiment": "OS-Genesis-7B with history enabled",
        "total_tasks": len(results),
        "overall_success_rate": success_rate,
        "single_step_success_rate": single_step_rate,
        "multi_step_success_rate": multi_step_rate,
        "average_progress": avg_progress,
        "single_step_count": len(single_step_results),
        "multi_step_count": len(multi_step_results)
    }

    summary_path = os.path.join(args.output_dir, "OS_Genesis_7b_with_history_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OS-Genesis with history enabled (parallel).")

    parser.add_argument(
        "--jsonl_file",
        type=str,
        default="/evaluation/dataset/android_control_evaluation_std.jsonl",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/evaluation/result_ac_mp",
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use in call_mobile_agent_vllm."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel threads (API calls). Default: 4"
    )

    args = parser.parse_args()
    main(args)
