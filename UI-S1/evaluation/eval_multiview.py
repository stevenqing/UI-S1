"""
Multi-View evaluation on AndroidControl trajectories.

Based on eval_qwenvl.py. Adds --mode (baseline|multiview), --variant (A/B/C/D),
--api_url, --save_debug, and --resume flags.
"""

import argparse
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image
from qwenvl_utils import (call_mobile_agent_vllm,
                          evaluate_android_control_action, find_last_image_ele,
                          message_translate, image_to_data_url)
from multiview_utils import MultiViewPipeline, VariantGPipeline, \
    _extract_images_from_content, _ensure_content_list

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from x.data.agent.json import JsonFormat
from x.qwen.data_format import slim_messages
import re


def robust_parse_response(fm, model_response):
    """Parse model response, with fallback for malformed <action> tags."""
    try:
        return fm.parse_response(model_response)
    except Exception:
        pass
    # Fallback: find JSON object in the response
    match = re.search(r'\{[^{}]*"action"\s*:', model_response)
    if match:
        # Find the complete JSON object
        start = match.start()
        brace_count = 0
        for i in range(start, len(model_response)):
            if model_response[i] == '{':
                brace_count += 1
            elif model_response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = model_response[start:i+1]
                    action_content = json.loads(json_str)
                    return {'think': None, 'action': json_str,
                            'action_content': action_content}
    raise ValueError(f"Cannot parse model response: {model_response[:200]}")

# Global read-only state, initialised by main thread
RAW_SPACE = None
fm = None
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


def _messages_to_openai(messages):
    """Convert Qwen-format messages to OpenAI format with base64 images."""
    translated, screenshot_list = message_translate(messages, to_format='openai')
    ptr = 0
    for msg in translated:
        for content in msg['content']:
            if 'image_url' in content:
                url = image_to_data_url(Image.open(screenshot_list[ptr]))
                content['image_url']['url'] = url
                ptr += 1
    assert ptr == len(screenshot_list)
    return translated


def process_line(line, args, view_executor=None):
    global fm

    num_steps = len(line['steps'])
    state = None
    model_response = None
    step_id = 0
    task_success = False
    fixed_line = fix_line(line)

    step_details = []

    # Create a per-trajectory pipeline for multiview mode
    pipeline = None
    g_pipeline = None
    if args.mode == 'multiview':
        if args.variant == 'G':
            # Variant G: D2-style single-turn, needs output_format and action_space
            # We extract these from the system prompt by doing a dummy gen_next_round
            dummy_state = fm.gen_next_round(
                fixed_line, None, previous_model_response=None)
            if dummy_state:
                sys_text = ""
                for msg in dummy_state['messages']:
                    if msg.get('role') == 'system':
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            sys_text = content
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and 'text' in item:
                                    sys_text = item['text']
                                    break
                        break
                output_format = ""
                action_space = ""
                if "## Output Format" in sys_text and "## Action Space" in sys_text:
                    parts = sys_text.split("## Action Space")
                    action_space = parts[1].split("## Note")[0].strip() if len(parts) > 1 else ""
                    fmt_parts = sys_text.split("## Output Format")
                    if len(fmt_parts) > 1:
                        output_format = fmt_parts[1].split("## Action Space")[0].strip()
                g_pipeline = VariantGPipeline(
                    args.model_name, args.api_url,
                    output_format=output_format, action_space=action_space)
                # Reset state for actual evaluation
                state = None
        else:
            pipeline = MultiViewPipeline(
                args.model_name, args.api_url, variant=args.variant)

    try:
        while step_id < num_steps:
            current_check_pam = fixed_line['steps'][step_id]['check_options']
            state = fm.gen_next_round(
                fixed_line, state, previous_model_response=model_response)
            if state is None:
                break

            messages = state['messages']
            messages = slim_messages(
                messages=messages,
                num_image_limit=args.n_history_image_limit)

            current_image_ele, width, height, resized_width, resized_height = \
                find_last_image_ele(messages)

            debug_info = None

            if args.mode == 'multiview' and args.variant == 'G':
                # Variant G: extract screenshot from messages, use single-turn pipeline
                openai_msgs = _messages_to_openai(messages)
                # Extract images from the last user message
                image_content = []
                for msg in reversed(openai_msgs):
                    if msg["role"] == "user":
                        image_content = _extract_images_from_content(
                            msg.get("content", []))
                        break
                model_response, debug_info = g_pipeline.step(
                    line['goal'], image_content, executor=view_executor)
            elif args.mode == 'multiview':
                # Convert to OpenAI format for multiview pipeline
                openai_msgs = _messages_to_openai(messages)
                model_response, debug_info = pipeline.step(
                    openai_msgs, executor=view_executor)
            else:
                model_response = call_mobile_agent_vllm(
                    messages=messages, model_name=args.model_name)

            pred_action = robust_parse_response(fm, model_response)
            type_match, extract_match = evaluate_android_control_action(
                pred_action['action_content'],
                current_check_pam,
                width, height,
                resized_width, resized_height)

            step_info = {
                "step_id": step_id,
                "type_match": bool(type_match),
                "extract_match": bool(extract_match),
                "action_type": current_check_pam.get('action', ''),
            }
            if args.save_debug and debug_info:
                step_info["visual_analysis"] = debug_info.get("visual_analysis", "")
                step_info["task_analysis"] = debug_info.get("task_analysis", "")
                step_info["model_response"] = model_response[:500]
                step_info["variant"] = debug_info.get("variant", "")
            step_details.append(step_info)

            if not extract_match:
                break

            step_id += 1

        task_success = (step_id == num_steps)

    except Exception as e:
        print(f"Error processing goal '{line['goal']}': {e}")
        import traceback
        traceback.print_exc()
        task_success = False
        step_id = 0

    result = {
        "goal": line['goal'],
        "num_steps": num_steps,
        "task_success": bool(task_success),
        "final_step_id": int(step_id),
        "mode": args.mode,
        "variant": getattr(args, 'variant', ''),
    }
    if args.save_debug:
        result["step_details"] = step_details

    # Thread-safe write
    suffix = _get_suffix(args)
    model_short = os.path.basename(args.model_name.rstrip('/'))
    with result_lock:
        result_path = os.path.join(
            args.output_dir, f"{model_short}{suffix}.jsonl")
        with open(result_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return result


def _get_suffix(args):
    if args.mode == "baseline":
        return "_baseline"
    return f"_multiview_{args.variant}"


def _load_completed_goals(output_dir, model_name, args):
    """Load goals already evaluated (for --resume)."""
    suffix = _get_suffix(args)
    model_short = os.path.basename(model_name.rstrip('/'))
    path = os.path.join(output_dir, f"{model_short}{suffix}.jsonl")
    completed = set()
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add(r['goal'])
                except Exception:
                    pass
    return completed


def main(args):
    global RAW_SPACE, fm

    from x.data.agent.space.std_space import RAW_SPACE as _RAW_SPACE
    RAW_SPACE = _RAW_SPACE
    if args.no_thought:
        fm = JsonFormat(RAW_SPACE, add_thought=False)
    else:
        fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    std_data = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            std_data.append(json.loads(line))

    # Resume support
    if args.resume:
        completed = _load_completed_goals(
            args.output_dir, args.model_name, args)
        before = len(std_data)
        std_data = [d for d in std_data if d['goal'] not in completed]
        print(f"Resume: skipping {before - len(std_data)} already completed, "
              f"{len(std_data)} remaining.")
    else:
        # Clear previous results if not resuming
        suffix = _get_suffix(args)
        model_short = os.path.basename(args.model_name.rstrip('/'))
        result_path = os.path.join(
            args.output_dir, f"{model_short}{suffix}.jsonl")
        if os.path.exists(result_path):
            os.remove(result_path)

    print(f"Loaded {len(std_data)} tasks. Mode: {args.mode}. "
          f"Variant: {args.variant}. Starting parallel evaluation...")

    # Inner executor for view parallelism (only used in multiview mode)
    view_executor = None
    if args.mode == 'multiview':
        view_executor = ThreadPoolExecutor(max_workers=2)

    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_line = {
                executor.submit(process_line, line, args, view_executor): line
                for line in std_data
            }
            for future in as_completed(future_to_line):
                try:
                    result = future.result()
                    results.append(result)
                    if len(results) % 50 == 0:
                        _print_interim(results)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
    finally:
        if view_executor is not None:
            view_executor.shutdown(wait=False)

    # If resuming, also load previous results for final stats
    if args.resume:
        suffix = _get_suffix(args)
        model_short = os.path.basename(args.model_name.rstrip('/'))
        result_path = os.path.join(
            args.output_dir, f"{model_short}{suffix}.jsonl")
        results = []
        with open(result_path, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass

    _print_final(results, args)


def _print_interim(results):
    n = len(results)
    success = sum(1 for r in results if r["task_success"])
    print(f"  Progress: {n} done, TSR={success/n*100:.1f}%")


def _print_final(results, args):
    if not results:
        print("No results.")
        return

    n = len(results)
    success_count = sum(1 for r in results if r["task_success"])
    success_rate = success_count / n * 100
    avg_progress = sum(
        r["final_step_id"] / r["num_steps"] for r in results) / n

    print(f"\n{'='*50}")
    print(f"Evaluation completed. Mode: {args.mode} Variant: {args.variant}")
    print(f"{'='*50}")
    print(f"Total: {n}")
    print(f"Success Rate (TSR): {success_rate:.2f}% ({success_count}/{n})")
    print(f"Average Progress: {avg_progress:.4f}")

    # Per-step type breakdown from debug info
    if args.save_debug:
        type_stats = {}
        for r in results:
            for s in r.get("step_details", []):
                atype = s.get("action_type", "unknown")
                if atype not in type_stats:
                    type_stats[atype] = {"total": 0, "type_match": 0,
                                         "extract_match": 0}
                type_stats[atype]["total"] += 1
                if s.get("type_match"):
                    type_stats[atype]["type_match"] += 1
                if s.get("extract_match"):
                    type_stats[atype]["extract_match"] += 1
        if type_stats:
            print("\nPer-action-type breakdown:")
            for atype, st in sorted(type_stats.items()):
                t = st["total"]
                print(f"  {atype}: type_match={st['type_match']}/{t} "
                      f"({st['type_match']/t*100:.1f}%)  "
                      f"extract_match={st['extract_match']}/{t} "
                      f"({st['extract_match']/t*100:.1f}%)")

    # Save summary
    summary = {
        "mode": args.mode,
        "variant": args.variant,
        "model_name": args.model_name,
        "total": n,
        "success_rate": success_rate,
        "avg_progress": avg_progress,
    }
    suffix = _get_suffix(args)
    model_short = os.path.basename(args.model_name.rstrip('/'))
    summary_path = os.path.join(
        args.output_dir, f"{model_short}{suffix}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-View evaluation on AndroidControl trajectories.")

    parser.add_argument(
        "--jsonl_file", type=str,
        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/android_control_evaluation_fixed.jsonl",
        help="Path to the input JSONL file.")
    parser.add_argument(
        "--output_dir", type=str,
        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/multiview",
        help="Directory to save evaluation results.")
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Model name served by vLLM.")
    parser.add_argument(
        "--mode", type=str, default="multiview",
        choices=["multiview", "baseline"],
        help="Evaluation mode: multiview (3-pass) or baseline (single pass).")
    parser.add_argument(
        "--variant", type=str, default="B",
        choices=["A", "B", "C", "D", "E", "F", "G"],
        help="Multi-view variant: A-D (analysis injection), "
             "E (action voting), F (ultra-concise hint), "
             "G (D2-style single-turn state tracker).")
    parser.add_argument(
        "--api_url", type=str, default="http://localhost:8000/v1",
        help="vLLM API endpoint URL.")
    parser.add_argument(
        "--n_history_image_limit", type=int, default=2,
        help="Maximum number of historical images to keep.")
    parser.add_argument(
        "--max_workers", type=int, default=4,
        help="Number of parallel threads for episodes.")
    parser.add_argument(
        "--save_debug", action="store_true",
        help="Save per-step view analyses in output.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-evaluated episodes (append to existing output).")
    parser.add_argument(
        "--no_thought", action="store_true",
        help="Use action-only format (no <think> tags). For models trained without thought.")

    args = parser.parse_args()
    main(args)
