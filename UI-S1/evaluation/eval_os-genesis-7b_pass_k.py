import argparse
import copy
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from agentcpm_utils import map_action_space2qwenvl
from os_genesis_utils import os_gensis_2minicpm
from PIL import Image
from qwenvl_utils import evaluate_android_control_action
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

result_lock = Lock()  # For thread-safe file writing


def predict_with_sampling(model_name, instruction, low_instruction, history, image, temperature=0.7, top_k=50):
    """Predict with temperature sampling for pass@k evaluation."""
    import base64
    import time
    import traceback
    from io import BytesIO
    from openai import OpenAI

    END_POINT = "http://localhost:8000/v1"

    def image_to_data_url(img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64_str}"

    a11y_tree = ""
    text = f"""You are a GUI task expert, I will provide you with a high-level instruction, an action history, a screenshot with its corresponding accessibility tree.

        High-level instruction: {instruction}
        Action history: {history}
        Accessibility tree: {a11y_tree}

        Please generate the low-level thought and action for the next step."""

    url = image_to_data_url(image)
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text}
        ]}
    ]

    max_retries = 5
    for i in range(max_retries):
        try:
            bot = OpenAI(
                api_key="EMPTY",
                base_url=END_POINT,
                timeout=300
            )
            # Use temperature sampling for pass@k
            kwargs = {
                'temperature': temperature,
                'extra_body': {"top_k": top_k}
            }
            chat_completion = bot.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                **kwargs
            )
            output = chat_completion.choices[0].message.content
            return output
        except:
            traceback.print_exc()
            print(f"Network Error (retry {i+1}/{max_retries}):")
            time.sleep(2)

    print(f"Max retries ({max_retries}) exhausted, skipping task")
    return ""


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


def evaluate_single_sample(line, args, sample_id):
    """Evaluate a single sample for pass@k."""
    num_steps = len(line['steps'])
    step_id = 0
    task_success = False
    fixed_line = fix_line(copy.deepcopy(line))
    history_list = []

    try:
        while step_id < num_steps:
            step = fixed_line['steps'][step_id]
            image_path = step['screenshot']
            low_instruction = step['step_instruction'] if 'step_instruction' in step else ''
            current_check_pam = step['check_options']
            history = ""  # OS-Genesis doesn't use history
            image = Image.open(image_path)
            width, height = image.size

            # Use temperature sampling
            model_response = predict_with_sampling(
                model_name=args.model_name,
                instruction=line['goal'],
                low_instruction='',
                history=history,
                image=image,
                temperature=args.temperature,
                top_k=50
            )

            action_minicpm = os_gensis_2minicpm(model_response)
            history_list.append(low_instruction)
            # OS-Genesis outputs absolute pixel coordinates
            pred_action = map_action_space2qwenvl(action_minicpm, [width, height], coordinate_format="absolute")

            type_match, extract_match = evaluate_android_control_action(
                pred_action,
                current_check_pam,
                width, height,
                width, height,
                ignore_actions=[]
            )

            if not extract_match:
                break

            step_id += 1

        task_success = (step_id == num_steps)

    except Exception as e:
        print(f"Error processing goal '{line['goal']}' (sample {sample_id}): {e}")
        traceback.print_exc()
        task_success = False
        step_id = 0

    return {
        "sample_id": sample_id,
        "task_success": task_success,
        "final_step_id": step_id,
    }


def process_line_pass_k(line, args):
    """Process a line with K samples for pass@k evaluation."""
    num_steps = len(line['steps'])
    fixed_line = fix_line(copy.deepcopy(line))

    # Run K samples
    sample_results = []
    for sample_id in range(args.num_samples):
        sample_result = evaluate_single_sample(fixed_line, args, sample_id)
        sample_results.append(sample_result)

        # Early exit if we already have a success (for efficiency)
        if sample_result['task_success']:
            print(f"Task '{line['goal'][:50]}...' succeeded at sample {sample_id + 1}/{args.num_samples}")
            break

    # pass@k: task succeeds if ANY sample succeeds
    any_success = any(r['task_success'] for r in sample_results)
    num_successes = sum(1 for r in sample_results if r['task_success'])
    best_progress = max(r['final_step_id'] for r in sample_results)

    result = {
        "goal": line['goal'],
        "num_steps": num_steps,
        "num_samples": len(sample_results),
        "num_successes": num_successes,
        "pass_k": any_success,
        "best_progress": best_progress,
        "sample_results": sample_results,
    }

    # Thread-safe write
    with result_lock:
        result_path = os.path.join(args.output_dir, f"OS_Genesis_7b_pass_k{args.num_samples}.jsonl")
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

    print(f"=" * 60)
    print(f"OS-Genesis-7B Pass@{args.num_samples} Evaluation")
    print(f"Model: {args.model_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Tasks: {len(std_data)}")
    print(f"=" * 60)

    # Clear previous results file
    result_path = os.path.join(args.output_dir, f"OS_Genesis_7b_pass_k{args.num_samples}.jsonl")
    if os.path.exists(result_path):
        os.remove(result_path)

    # Parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_line = {executor.submit(process_line_pass_k, line, args): line for line in std_data}
        for future in as_completed(future_to_line):
            try:
                result = future.result()
                results.append(result)

                # Print progress
                completed = len(results)
                pass_k_count = sum(1 for r in results if r["pass_k"])
                print(f"Progress: {completed}/{len(std_data)} | Pass@{args.num_samples}: {pass_k_count}/{completed} ({100*pass_k_count/completed:.1f}%)")
            except Exception as e:
                print(f"Task generated an exception: {e}")

    # Calculate final metrics
    pass_k_count = sum(1 for r in results if r["pass_k"])
    pass_k_rate = pass_k_count / len(results) * 100 if results else 0

    # Also calculate pass@1 for comparison (first sample only)
    pass_1_count = sum(1 for r in results if r["sample_results"][0]["task_success"])
    pass_1_rate = pass_1_count / len(results) * 100 if results else 0

    avg_best_progress = sum(r["best_progress"] / r['num_steps'] for r in results) / len(results) if results else 0.0
    avg_successes = sum(r["num_successes"] for r in results) / len(results) if results else 0.0

    print(f"\n{'=' * 60}")
    print(f"OS-Genesis-7B Pass@{args.num_samples} Evaluation Completed")
    print(f"{'=' * 60}")
    print(f"Pass@1 Rate: {pass_1_rate:.2f}% ({pass_1_count}/{len(results)})")
    print(f"Pass@{args.num_samples} Rate: {pass_k_rate:.2f}% ({pass_k_count}/{len(results)})")
    print(f"Average Best Progress: {avg_best_progress:.2f}")
    print(f"Average Successes per Task: {avg_successes:.2f}")
    print(f"Results saved to: {result_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass@K evaluation for OS-Genesis-7B on Android control tasks.")

    parser.add_argument(
        "--jsonl_file",
        type=str,
        default="/evaluation/dataset/android_control_evaluation_std.jsonl",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/evaluation/results/pass_k",
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel threads. Default: 4"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples per task for pass@k. Default: 8"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Default: 0.7"
    )

    args = parser.parse_args()
    main(args)
