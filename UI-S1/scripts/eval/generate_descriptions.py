"""
Offline pi_V description generation for training data.

For each step that has s_{t+1}, generate a text description of s_{t+1}
using the base model. Output: new JSONL with desc_t1 field added to each step.

Usage:
    python generate_descriptions.py \
        --input_jsonl datasets/ui_s1_dataset/ui_s1_train.jsonl \
        --output_jsonl datasets/ui_s1_dataset/ui_s1_train_with_desc.jsonl \
        --model_name qwen25vl_7b_base \
        --max_workers 128
"""

import argparse
import base64
import json
import os
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Lock

from openai import OpenAI
from PIL import Image

write_lock = Lock()

DESCRIBE_PROMPT = """Describe the current state of this mobile app screen in detail.
1. What app is this? What screen/page is shown?
2. List the main UI elements visible (buttons, text fields, lists, icons, images).
3. Note any text content displayed (titles, labels, input field values, list items).
4. Describe any active states: open keyboards, dialogs, menus, loading indicators, selected items.
Be specific and concise."""


def encode_screenshot(path):
    img = Image.open(path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def call_vllm_describe(client, model_name, image_path, max_tokens=512):
    data_url = encode_screenshot(image_path)
    content = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": DESCRIBE_PROMPT},
    ]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                extra_body={"top_k": 1},
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                import time; time.sleep(3)
            else:
                raise


def main(args):
    # Load dataset
    lines = []
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    print(f"Loaded {len(lines)} episodes")

    # Collect all unique screenshots that need descriptions
    # (s_{t+1} for each step, i.e. step[i+1].screenshot)
    screenshot_to_desc = {}
    screenshots_to_generate = []

    for ep_idx, line in enumerate(lines):
        for si in range(len(line['steps'])):
            if si + 1 < len(line['steps']):
                ss_path = line['steps'][si + 1]['screenshot']
                if ss_path not in screenshot_to_desc:
                    screenshot_to_desc[ss_path] = None  # placeholder
                    screenshots_to_generate.append(ss_path)

    print(f"Unique screenshots to describe: {len(screenshots_to_generate)}")

    # Generate descriptions
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=600)

    n_success = 0
    n_error = 0

    def describe_screenshot(ss_path):
        try:
            desc = call_vllm_describe(client, args.model_name, ss_path)
            return ss_path, desc, None
        except Exception as e:
            traceback.print_exc()
            return ss_path, None, str(e)

    print(f"\nGenerating descriptions with max_workers={args.max_workers}...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(describe_screenshot, ss): ss
                   for ss in screenshots_to_generate}
        for i, future in enumerate(as_completed(futures)):
            ss_path, desc, err = future.result()
            if desc is not None:
                screenshot_to_desc[ss_path] = desc
                n_success += 1
            else:
                n_error += 1
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i+1}/{len(screenshots_to_generate)}, "
                      f"success={n_success}, error={n_error}")

    print(f"\nDone: {n_success} success, {n_error} error "
          f"out of {len(screenshots_to_generate)} screenshots")

    # Add desc_t1 to each step
    n_steps_with_desc = 0
    n_steps_without_desc = 0
    n_last_steps = 0

    for line in lines:
        for si in range(len(line['steps'])):
            if si + 1 < len(line['steps']):
                ss_path = line['steps'][si + 1]['screenshot']
                desc = screenshot_to_desc.get(ss_path)
                if desc:
                    line['steps'][si]['desc_t1'] = desc
                    n_steps_with_desc += 1
                else:
                    line['steps'][si]['desc_t1'] = None
                    n_steps_without_desc += 1
            else:
                # Last step: no s_{t+1}
                line['steps'][si]['desc_t1'] = None
                n_last_steps += 1

    print(f"\nSteps with desc_t1: {n_steps_with_desc}")
    print(f"Steps without desc_t1 (error): {n_steps_without_desc}")
    print(f"Last steps (no s_{{t+1}}): {n_last_steps}")

    # Write output
    os.makedirs(os.path.dirname(args.output_jsonl) or '.', exist_ok=True)
    with open(args.output_jsonl, 'w') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f"\nOutput: {args.output_jsonl}")

    # Also save screenshot->desc mapping for reuse
    mapping_file = args.output_jsonl.replace('.jsonl', '_desc_mapping.json')
    mapping = {k: v for k, v in screenshot_to_desc.items() if v is not None}
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1)
    print(f"Description mapping: {mapping_file} ({len(mapping)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pi_V descriptions for training data s_{t+1}")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=128)
    args = parser.parse_args()
    main(args)
