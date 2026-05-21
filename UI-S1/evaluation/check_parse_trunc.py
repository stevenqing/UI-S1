#!/usr/bin/env python3
"""Check how many parse errors in our analysis are artifacts of 500-char save truncation."""
import json, re, glob

def parse_action(response):
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if not m:
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if not m:
        m = re.search(r'(\{"function".*?\})', response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("function"), data.get("args", {}), True
        except json.JSONDecodeError:
            pass
    return None, None, False

for model, base_path in [
    ("coop_v3_ep2", "train_GUI_360/GUI-360-eval/results/cooperative_thought_v3_ep2/action_prediction"),
    ("svd_r256", "train_GUI_360/GUI-360-eval/results/svd_lora_r256_same_pipeline/action_prediction"),
]:
    total_fail = 0
    parse_fail = 0
    trunc_parse_fail = 0
    real_parse_fail = 0

    for shard in range(4):
        files = glob.glob(f"{base_path}/results_shard{shard}_*.json")
        with open(files[0]) as f:
            data = json.load(f)
        for item in data:
            if item["success"] or "response" not in item:
                continue
            total_fail += 1
            resp = item["response"]
            fn, args, ok = parse_action(resp)
            if not ok:
                parse_fail += 1
                if len(resp) >= 499:
                    trunc_parse_fail += 1
                else:
                    real_parse_fail += 1

    print(f"=== {model} ===")
    print(f"  Total failures: {total_fail}")
    print(f"  Parse errors in post-hoc analysis: {parse_fail}")
    print(f"    Due to 500-char save truncation (len>=499): {trunc_parse_fail}")
    print(f"    Real parse errors (len<499): {real_parse_fail}")
    print()
