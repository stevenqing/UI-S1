import argparse
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image
from qwenvl_utils import evaluate_android_control_action, find_last_image_ele
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from x.data.agent.json import JsonFormat
from x.qwen.data_format import slim_messages

# Global variables (read-only), initialized by main thread
RAW_SPACE = None
fm = None
result_lock = Lock()  # For thread-safe file writing


def call_mobile_agent_vllm_with_sampling(messages, model_name, temperature=0.7, top_k=50):
    """Call vLLM API with temperature sampling for pass@k evaluation."""
    import base64
    import time
    import traceback
    from io import BytesIO
    from openai import OpenAI

    END_POINT = "http://localhost:8000/v1"

    def image_to_data_url(image):
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64_str}"

    # Convert messages to OpenAI format
    messages_openai = copy.deepcopy(messages)
    screenshot_list = []
    for msg in messages_openai:
        if isinstance(msg['content'], str):
            msg['content'] = [msg['content']]
        new_contents = []
        for content in msg['content']:
            if isinstance(content, str):
                new_contents.append({"type": "text", 'text': content})
            elif 'text' in content:
                new_contents.append({"type": "text", 'text': content['text']})
            elif 'image' in content:
                screenshot_list.append(content['image'])
                new_contents.append({"type": "image_url", "image_url": {"url": content['image']}})
            else:
                raise NotImplementedError
        msg['content'] = new_contents

    # Replace image paths with data URLs
    screenshot_ptr = 0
    for msg in messages_openai:
        for content in msg['content']:
            if 'image_url' in content:
                url = image_to_data_url(Image.open(screenshot_list[screenshot_ptr]))
                content['image_url']['url'] = url
                screenshot_ptr += 1

    assert screenshot_ptr == len(screenshot_list)

    output = ''
    max_retries = 5
    for i in range(max_retries):
        try:
            bot = OpenAI(
                api_key="EMPTY",
                base_url=END_POINT,
                timeout=600
            )
            # Use temperature sampling for pass@k
            kwargs = {
                'temperature': temperature,
                'extra_body': {"top_k": top_k}
            }
            chat_completion = bot.chat.completions.create(
                model=model_name,
                messages=messages_openai,
                **kwargs
            )
            output = chat_completion.choices[0].message.content
            return output
        except Exception as e:
            traceback.print_exc()
            print(f"Network Error (attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(5)
            else:
                print(f"Max retries ({max_retries}) exceeded, skipping this request.")
                return ""
    return output


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
    global fm

    num_steps = len(line['steps'])
    state = None
    model_response = None
    step_id = 0
    task_success = False
    fixed_line = fix_line(copy.deepcopy(line))

    try:
        while step_id < num_steps:
            current_check_pam = fixed_line['steps'][step_id]['check_options']
            state = fm.gen_next_round(fixed_line, state, previous_model_response=model_response)
            if state is None:
                break

            messages = state['messages']
            messages = slim_messages(messages=messages, num_image_limit=args.n_history_image_limit)
            current_image_ele, width, height, resized_width, resized_height = find_last_image_ele(messages)

            # Use temperature sampling
            model_response = call_mobile_agent_vllm_with_sampling(
                messages=messages,
                model_name=args.model_name,
                temperature=args.temperature,
                top_k=50
            )

            pred_action = fm.parse_response(model_response)
            type_match, extract_match = evaluate_android_control_action(
                pred_action['action_content'],
                current_check_pam,
                width, height,
                resized_width, resized_height
            )

            if not extract_match:
                break

            step_id += 1

        task_success = (step_id == num_steps)

    except Exception as e:
        print(f"Error processing goal '{line['goal']}' (sample {sample_id}): {e}")
        task_success = False
        step_id = 0

    return {
        "sample_id": sample_id,
        "task_success": task_success,
        "final_step_id": step_id,
    }


def process_line_pass_k(line, args):
    """Process a line with K samples for pass@k evaluation."""
    global fm

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
        result_path = os.path.join(args.output_dir, f"{args.model_name}_pass_k{args.num_samples}.jsonl")
        with open(result_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return result


def main(args):
    global RAW_SPACE, fm

    # Initialize global read-only components (in main thread)
    from x.data.agent.space.std_space import RAW_SPACE as _RAW_SPACE
    RAW_SPACE = _RAW_SPACE
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Read data
    std_data = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            std_data.append(json.loads(line))

    print(f"=" * 60)
    print(f"Pass@{args.num_samples} Evaluation")
    print(f"Model: {args.model_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Tasks: {len(std_data)}")
    print(f"=" * 60)

    # Clear previous results file
    result_path = os.path.join(args.output_dir, f"{args.model_name}_pass_k{args.num_samples}.jsonl")
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
    print(f"Pass@{args.num_samples} Evaluation Completed")
    print(f"{'=' * 60}")
    print(f"Pass@1 Rate: {pass_1_rate:.2f}% ({pass_1_count}/{len(results)})")
    print(f"Pass@{args.num_samples} Rate: {pass_k_rate:.2f}% ({pass_k_count}/{len(results)})")
    print(f"Average Best Progress: {avg_best_progress:.2f}")
    print(f"Average Successes per Task: {avg_successes:.2f}")
    print(f"Results saved to: {result_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass@K evaluation for mobile agent on Android control tasks.")

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
        "--n_history_image_limit",
        type=int,
        default=2,
        help="Maximum number of historical images to keep. Default: 2"
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
