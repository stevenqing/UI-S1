#!/usr/bin/env python3
"""
Corrected error analysis v2: uses success/failure from eval (ground truth),
then re-classifies ONLY the failure reason from saved responses.

Key fix: for responses truncated at 500 chars, use a lenient parser that
doesn't require </tool_call> closing tag. This eliminates the false
parse_error artifacts.

Usage:
  python evaluation/analysis_wrong_coordinate_v2.py \
      --test_data_root datasets/GUI-360/test \
      --result_dirs coop_v3_ep2=train_GUI_360/GUI-360-eval/results/cooperative_thought_v3_ep2/action_prediction \
                    svd_lora_r256=train_GUI_360/GUI-360-eval/results/svd_lora_r256_same_pipeline/action_prediction \
      --output_dir evaluation/analysis_results_v2
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ── Load ground truth ─────────────────────────────────────────────

def load_ground_truth(test_data_root):
    """Load test samples matching eval script's logic exactly."""
    data_path = os.path.join(test_data_root, "data")
    gt = {}

    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for category in sorted(os.listdir(domain_path)):
            cat_path = os.path.join(domain_path, category, "success")
            if not os.path.exists(cat_path):
                continue
            for jsonl_file in sorted(os.listdir(cat_path)):
                if not jsonl_file.endswith(".jsonl"):
                    continue
                fpath = os.path.join(cat_path, jsonl_file)
                all_steps = []
                with open(fpath) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        all_steps.append({"line_num": line_num, "data": data})

                for i, step_info in enumerate(all_steps):
                    data = step_info["data"]
                    step = data.get("step", {})
                    tags = step.get("tags", [])
                    if "action_prediction" not in tags:
                        continue
                    action = step.get("action", {})
                    if action.get("function") == "drag" or not action.get("rectangle"):
                        continue

                    sample_id = f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}_{step_info['line_num']}"

                    # Transform args to match eval format
                    args = dict(action.get("args", {}))
                    args.pop("x", None)
                    args.pop("y", None)
                    if action.get("coordinate_x") is not None:
                        args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]

                    gt[sample_id] = {
                        "domain": domain,
                        "category": category,
                        "step_id": data.get("step_id", step_info["line_num"]),
                        "total_steps": data.get("total_steps", 0),
                        "request": data.get("request", ""),
                        "thought_gt": step.get("thought", ""),
                        "function": action.get("function", ""),
                        "args": args,
                        "rectangle": action.get("rectangle", {}),
                        "gt_x": action.get("coordinate_x"),
                        "gt_y": action.get("coordinate_y"),
                        "control_text": action.get("control_test", ""),
                        "tags": tags,
                    }

    print(f"Loaded {len(gt)} ground truth samples")
    return gt


# ── Load prediction results ───────────────────────────────────────

def load_results(result_dir):
    results = []
    for fname in sorted(os.listdir(result_dir)):
        if fname.startswith("results_") and fname.endswith(".json"):
            with open(os.path.join(result_dir, fname)) as f:
                data = json.load(f)
            results.extend(data)
    print(f"  Loaded {len(results)} predictions from {result_dir}")
    return results


# ── Lenient parser (handles 500-char truncation) ──────────────────

def parse_action_lenient(response):
    """Parse function/args from response, tolerant of truncation.

    Tries in order:
    1. Full <tool_call>{JSON}</tool_call>
    2. <tool_call>{JSON} (no closing tag — truncated)
    3. {"function": ...} anywhere
    4. Partial JSON reconstruction for truncated responses
    """
    if not response or not response.strip():
        return None, None, "empty"

    # Strategy 1: full tool_call tags
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("function"), data.get("args", {}), "full_parse"
        except json.JSONDecodeError:
            pass

    # Strategy 2: tool_call without closing tag (truncated)
    m = re.search(r'<tool_call>\s*(\{.*)', response, re.DOTALL)
    if m:
        json_str = m.group(1).strip()
        # Try parsing as-is
        try:
            data = json.loads(json_str)
            return data.get("function"), data.get("args", {}), "truncated_parse"
        except json.JSONDecodeError:
            pass
        # Try completing truncated JSON
        parsed = _try_complete_json(json_str)
        if parsed:
            return parsed.get("function"), parsed.get("args", {}), "reconstructed_parse"

    # Strategy 3: bare JSON object
    m = re.search(r'(\{"function".*?\})', response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("function"), data.get("args", {}), "bare_json"
        except json.JSONDecodeError:
            pass

    # Strategy 4: extract function name at minimum
    m = re.search(r'"function"\s*:\s*"([^"]*)"', response)
    fn = m.group(1) if m else None

    # Try to extract coordinate
    coord_m = re.search(r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)', response)
    if fn and coord_m:
        args = {"coordinate": [float(coord_m.group(1)), float(coord_m.group(2))]}
        # Try to extract other args
        for key in ["button", "double", "text", "keys"]:
            km = re.search(rf'"{key}"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|\d+)', response)
            if km:
                val = km.group(1)
                try:
                    args[key] = json.loads(val)
                except json.JSONDecodeError:
                    args[key] = val.strip('"')
        return fn, args, "regex_extract"
    elif fn:
        return fn, {}, "function_only"

    return None, None, "parse_error"


def _try_complete_json(json_str):
    """Try to complete a truncated JSON string by adding closing brackets."""
    # Count open/close brackets
    for suffix in ["}", "]}", "]]}", '"]}', '"]]}', '"}']:
        try:
            return json.loads(json_str + suffix)
        except json.JSONDecodeError:
            continue
    return None


# ── Coordinate helpers ────────────────────────────────────────────

def coord_in_rect(x, y, rect):
    return (rect.get("left", 0) <= x <= rect.get("right", 0) and
            rect.get("top", 0) <= y <= rect.get("bottom", 0))


def rect_center(rect):
    cx = (rect.get("left", 0) + rect.get("right", 0)) / 2
    cy = (rect.get("top", 0) + rect.get("bottom", 0)) / 2
    return cx, cy


def euclidean_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ── Error classification ─────────────────────────────────────────

def classify_error(pred_fn, pred_args, parse_status, gt_info):
    """Classify a failed prediction into detailed error categories."""
    extra = {}

    if parse_status in ("empty", "parse_error"):
        return "parse_error", extra

    gt_fn = gt_info["function"]
    gt_args = gt_info["args"]
    rect = gt_info["rectangle"]

    if pred_fn is None or pred_fn == "":
        return "empty_function", extra

    if pred_fn != gt_fn:
        extra["pred_function"] = pred_fn
        extra["gt_function"] = gt_fn
        return "wrong_function", extra

    # Function matches — check coordinate first
    if "coordinate" in gt_args:
        if not pred_args or "coordinate" not in pred_args:
            return "missing_coordinate", extra
        try:
            px = float(pred_args["coordinate"][0])
            py = float(pred_args["coordinate"][1])
        except (TypeError, ValueError, IndexError):
            return "bad_coordinate_format", extra

        in_rect = coord_in_rect(px, py, rect)
        gt_cx, gt_cy = rect_center(rect)
        dist = euclidean_dist(px, py, gt_cx, gt_cy)
        extra["pred_x"] = px
        extra["pred_y"] = py
        extra["gt_cx"] = gt_cx
        extra["gt_cy"] = gt_cy
        extra["distance"] = dist
        extra["rect"] = rect

        if not in_rect:
            # Check if other args also wrong
            other_wrong = False
            for key in gt_args:
                if key == "coordinate":
                    continue
                if key not in pred_args or str(pred_args[key]).lower() != str(gt_args[key]).lower():
                    other_wrong = True
                    break
            if other_wrong:
                return "wrong_coordinate+wrong_args", extra
            return "wrong_coordinate", extra

        # Coordinate OK — check other args
        for key in gt_args:
            if key == "coordinate":
                continue
            if key not in pred_args:
                extra["missing_key"] = key
                return f"missing_arg_{key}", extra
            if str(pred_args[key]).lower() != str(gt_args[key]).lower():
                extra["key"] = key
                extra["pred_val"] = str(pred_args[key])
                extra["gt_val"] = str(gt_args[key])
                return f"wrong_arg_{key}", extra

        # Everything looks correct but eval said failure — could be
        # parse_status=function_only (we guessed args) or rounding
        return "unknown_match", extra

    else:
        # No coordinate in GT — check other args
        if not pred_args or not gt_args:
            return "wrong_args", extra
        for key in gt_args:
            if key not in pred_args:
                return f"missing_arg_{key}", extra
            if str(pred_args[key]).lower() != str(gt_args[key]).lower():
                return f"wrong_arg_{key}", extra
        return "unknown_match", extra


# ── UI region classification ──────────────────────────────────────

def classify_ui_region(gt_y, domain="excel"):
    if gt_y is None:
        return "unknown"
    if gt_y < 30:
        return "title_bar"
    elif gt_y < 130:
        return "ribbon"
    elif gt_y < 160 and domain == "excel":
        return "formula_bar"
    elif gt_y < 700:
        return "content_area"
    else:
        return "status_bar"


# ── Analysis ──────────────────────────────────────────────────────

def analyze_model(model_name, results, gt_dict):
    analysis = {
        "model": model_name,
        "total": 0, "success": 0, "fail": 0,
        "error_types": Counter(),
        "error_types_coarse": Counter(),  # coarse: wrong_coord, wrong_fn, wrong_args, parse_error
        "distances": [],
        "errors_by_region": defaultdict(Counter),
        "errors_by_function": defaultdict(Counter),
        "errors_by_step_pos": defaultdict(Counter),
        "errors_by_domain": defaultdict(Counter),
        "success_by_region": Counter(),
        "total_by_region": Counter(),
        "success_by_function": Counter(),
        "total_by_function": Counter(),
        "success_by_step_pos": Counter(),
        "total_by_step_pos": Counter(),
        "success_by_domain": Counter(),
        "total_by_domain": Counter(),
        "parse_statuses": Counter(),
        "wrong_coord_details": [],
        "wrong_fn_confusion": Counter(),  # (gt_fn, pred_fn) pairs
    }

    for item in results:
        sid = item["sample_id"]
        gt = gt_dict.get(sid)
        if gt is None:
            continue

        analysis["total"] += 1

        # Dimensions
        step_id = gt["step_id"]
        total_steps = gt["total_steps"]
        if total_steps > 0:
            ratio = step_id / total_steps
            step_pos = "early" if ratio <= 0.33 else ("mid" if ratio <= 0.66 else "late")
        else:
            step_pos = "unknown"

        region = classify_ui_region(gt["gt_y"], gt["domain"])
        gt_fn = gt["function"]
        domain = gt["domain"]

        analysis["total_by_region"][region] += 1
        analysis["total_by_function"][gt_fn] += 1
        analysis["total_by_step_pos"][step_pos] += 1
        analysis["total_by_domain"][domain] += 1

        if item["success"]:
            analysis["success"] += 1
            analysis["success_by_region"][region] += 1
            analysis["success_by_function"][gt_fn] += 1
            analysis["success_by_step_pos"][step_pos] += 1
            analysis["success_by_domain"][domain] += 1
            continue

        analysis["fail"] += 1

        # Re-classify error using lenient parser
        response = item.get("response", "")
        pred_fn, pred_args, parse_status = parse_action_lenient(response)
        analysis["parse_statuses"][parse_status] += 1

        error_type, extra = classify_error(pred_fn, pred_args, parse_status, gt)
        analysis["error_types"][error_type] += 1

        # Coarse category
        if error_type in ("parse_error", "empty_function", "bad_coordinate_format",
                          "missing_coordinate"):
            coarse = "format_error"
        elif error_type == "wrong_function":
            coarse = "wrong_function"
        elif error_type.startswith("wrong_coordinate"):
            coarse = "wrong_coordinate"
        elif error_type.startswith("wrong_arg_") or error_type.startswith("missing_arg_"):
            coarse = "wrong_args"
        else:
            coarse = "other"
        analysis["error_types_coarse"][coarse] += 1

        analysis["errors_by_region"][region][coarse] += 1
        analysis["errors_by_function"][gt_fn][coarse] += 1
        analysis["errors_by_step_pos"][step_pos][coarse] += 1
        analysis["errors_by_domain"][domain][coarse] += 1

        if error_type == "wrong_function" and pred_fn:
            analysis["wrong_fn_confusion"][(gt_fn, pred_fn)] += 1

        if coarse == "wrong_coordinate" and "distance" in extra:
            analysis["distances"].append(extra["distance"])
            thought_pred = ""
            m = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
            if m:
                thought_pred = m.group(1).strip()
            analysis["wrong_coord_details"].append({
                "sample_id": sid,
                "domain": domain,
                "step_id": step_id,
                "total_steps": total_steps,
                "step_pos": step_pos,
                "region": region,
                "gt_function": gt_fn,
                "pred_x": extra["pred_x"],
                "pred_y": extra["pred_y"],
                "gt_cx": extra["gt_cx"],
                "gt_cy": extra["gt_cy"],
                "distance": extra["distance"],
                "rect": extra["rect"],
                "thought_gt": gt["thought_gt"][:300],
                "thought_pred": thought_pred[:300],
                "control_text": gt["control_text"],
            })

    return analysis


def print_report(analyses, output_dir, all_results):
    os.makedirs(output_dir, exist_ok=True)
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    def row(*parts):
        line = "".join(parts)
        print(line)
        lines.append(line)

    p("=" * 80)
    p("ERROR ANALYSIS v2 (corrected for 500-char truncation)")
    p("=" * 80)

    # ── 1. Overall ──
    p("\n## 1. Overall Results\n")
    p(f"{'Model':<30} {'Total':>6} {'Success':>8} {'Rate':>7} {'Fail':>6}")
    p("-" * 60)
    for a in analyses:
        rate = 100 * a["success"] / a["total"] if a["total"] > 0 else 0
        p(f"{a['model']:<30} {a['total']:>6} {a['success']:>8} {rate:>6.1f}% {a['fail']:>6}")

    # ── 2. Coarse Error Breakdown ──
    p("\n## 2. Error Type Breakdown (Coarse)\n")
    coarse_types = ["wrong_coordinate", "wrong_function", "wrong_args", "format_error", "other"]
    parts = [f"{'Error Type':<25}"]
    for a in analyses:
        parts.append(f" {a['model']:>25}")
    row(*parts)
    p("-" * (25 + 26 * len(analyses)))

    for et in coarse_types:
        parts = [f"{et:<25}"]
        for a in analyses:
            count = a["error_types_coarse"].get(et, 0)
            pct = 100 * count / a["fail"] if a["fail"] > 0 else 0
            parts.append(f" {count:>8} ({pct:>5.1f}%)")
        row(*parts)

    # ── 2b. Fine-grained Error Breakdown ──
    p("\n## 2b. Error Type Breakdown (Fine-grained)\n")
    all_et = sorted(set().union(*(a["error_types"].keys() for a in analyses)),
                    key=lambda x: -max(a["error_types"].get(x, 0) for a in analyses))
    parts = [f"{'Error Type':<35}"]
    for a in analyses:
        parts.append(f" {a['model']:>20}")
    row(*parts)
    p("-" * (35 + 21 * len(analyses)))

    for et in all_et:
        parts = [f"{et:<35}"]
        for a in analyses:
            count = a["error_types"].get(et, 0)
            pct = 100 * count / a["fail"] if a["fail"] > 0 else 0
            parts.append(f" {count:>8} ({pct:>5.1f}%)")
        row(*parts)

    # ── 2c. Parse status (how the lenient parser handled responses) ──
    p("\n## 2c. Parse Status Distribution (failed samples)\n")
    all_ps = sorted(set().union(*(a["parse_statuses"].keys() for a in analyses)),
                    key=lambda x: -max(a["parse_statuses"].get(x, 0) for a in analyses))
    parts = [f"{'Parse Status':<25}"]
    for a in analyses:
        parts.append(f" {a['model']:>20}")
    row(*parts)
    p("-" * (25 + 21 * len(analyses)))

    for ps in all_ps:
        parts = [f"{ps:<25}"]
        for a in analyses:
            count = a["parse_statuses"].get(ps, 0)
            parts.append(f" {count:>8}")
        row(*parts)

    # ── 3. Wrong Coordinate Distance ──
    p("\n## 3. Wrong Coordinate Distance Distribution\n")
    dist_bins = [0, 20, 50, 100, 200, 500, float("inf")]
    bin_labels = ["0-20 (near-miss)", "20-50", "50-100", "100-200", "200-500", "500+ (far-miss)"]

    parts = [f"{'Distance bin':<25}"]
    for a in analyses:
        parts.append(f" {a['model']:>25}")
    row(*parts)
    p("-" * (25 + 26 * len(analyses)))

    for i, label in enumerate(bin_labels):
        parts = [f"{label:<25}"]
        for a in analyses:
            dists = a["distances"]
            count = sum(1 for d in dists if dist_bins[i] <= d < dist_bins[i + 1])
            pct = 100 * count / len(dists) if dists else 0
            parts.append(f" {count:>8} ({pct:>5.1f}%)")
        row(*parts)

    for a in analyses:
        dists = a["distances"]
        if dists:
            arr = np.array(dists)
            p(f"\n  {a['model']}: N={len(arr)}, mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
              f"p25={np.percentile(arr, 25):.1f}, p75={np.percentile(arr, 75):.1f}, "
              f"p90={np.percentile(arr, 90):.1f}")

    # ── 4. UI Region ──
    p("\n## 4. Success Rate by UI Region\n")
    all_regions = sorted(set().union(*(a["total_by_region"].keys() for a in analyses)))

    parts = [f"{'Region':<20}"]
    for a in analyses:
        parts.append(f" {a['model']:>25}")
    row(*parts)
    p("-" * (20 + 26 * len(analyses)))

    for region in all_regions:
        parts = [f"{region:<20}"]
        for a in analyses:
            total = a["total_by_region"].get(region, 0)
            succ = a["success_by_region"].get(region, 0)
            rate = 100 * succ / total if total > 0 else 0
            parts.append(f" {succ:>5}/{total:<5} ({rate:>5.1f}%)")
        row(*parts)

    p("\n  Error breakdown per region (coarse):")
    for region in all_regions:
        for a in analyses:
            total_fail_region = sum(a["errors_by_region"].get(region, {}).values())
            if total_fail_region > 10:
                et_parts = []
                for ct in coarse_types:
                    c = a["errors_by_region"].get(region, {}).get(ct, 0)
                    if c > 0:
                        et_parts.append(f"{ct}={c}")
                p(f"    {a['model']:<20} | {region}: {' '.join(et_parts)}")

    # ── 5. Function Type ──
    p("\n## 5. Success Rate by Function Type\n")
    all_fns = sorted(set().union(*(a["total_by_function"].keys() for a in analyses)),
                     key=lambda fn: -max(a["total_by_function"].get(fn, 0) for a in analyses))

    parts = [f"{'Function':<22}"]
    for a in analyses:
        parts.append(f" {a['model']:>25}")
    row(*parts)
    p("-" * (22 + 26 * len(analyses)))

    for fn in all_fns:
        parts = [f"{fn:<22}"]
        for a in analyses:
            total = a["total_by_function"].get(fn, 0)
            succ = a["success_by_function"].get(fn, 0)
            rate = 100 * succ / total if total > 0 else 0
            parts.append(f" {succ:>5}/{total:<5} ({rate:>5.1f}%)")
        row(*parts)

    # ── 5b. Function confusion matrix (top confusions) ──
    p("\n## 5b. Top Function Confusions (wrong_function)\n")
    for a in analyses:
        p(f"  {a['model']}:")
        top_conf = a["wrong_fn_confusion"].most_common(10)
        for (gt_fn, pred_fn), count in top_conf:
            p(f"    {gt_fn:>20} -> {pred_fn:<20} : {count}")

    # ── 6. Step Position ──
    p("\n## 6. Success Rate by Step Position\n")
    all_pos = ["early", "mid", "late"]

    parts = [f"{'Position':<20}"]
    for a in analyses:
        parts.append(f" {a['model']:>25}")
    row(*parts)
    p("-" * (20 + 26 * len(analyses)))

    for pos in all_pos:
        parts = [f"{pos:<20}"]
        for a in analyses:
            total = a["total_by_step_pos"].get(pos, 0)
            succ = a["success_by_step_pos"].get(pos, 0)
            rate = 100 * succ / total if total > 0 else 0
            parts.append(f" {succ:>5}/{total:<5} ({rate:>5.1f}%)")
        row(*parts)

    # ── 7. Domain ──
    p("\n## 7. Success Rate by Domain\n")
    all_domains = sorted(set().union(*(a["total_by_domain"].keys() for a in analyses)))

    parts = [f"{'Domain':<20}"]
    for a in analyses:
        parts.append(f" {a['model']:>25}")
    row(*parts)
    p("-" * (20 + 26 * len(analyses)))

    for dom in all_domains:
        parts = [f"{dom:<20}"]
        for a in analyses:
            total = a["total_by_domain"].get(dom, 0)
            succ = a["success_by_domain"].get(dom, 0)
            rate = 100 * succ / total if total > 0 else 0
            parts.append(f" {succ:>5}/{total:<5} ({rate:>5.1f}%)")
        row(*parts)

    # ── 8. Differential Analysis ──
    if len(analyses) == 2 and len(all_results) == 2:
        p("\n## 8. Differential Analysis\n")
        a0, a1 = analyses[0], analyses[1]
        r0 = {item["sample_id"]: item for item in all_results[0]}
        r1 = {item["sample_id"]: item for item in all_results[1]}

        common_ids = set(r0.keys()) & set(r1.keys())
        both_succ = sum(1 for sid in common_ids if r0[sid]["success"] and r1[sid]["success"])
        both_fail = sum(1 for sid in common_ids if not r0[sid]["success"] and not r1[sid]["success"])
        only_a0_fail = sum(1 for sid in common_ids if not r0[sid]["success"] and r1[sid]["success"])
        only_a1_fail = sum(1 for sid in common_ids if r0[sid]["success"] and not r1[sid]["success"])

        p(f"  Common samples: {len(common_ids)}")
        p(f"  Both succeed:    {both_succ} ({100*both_succ/len(common_ids):.1f}%)")
        p(f"  Both fail:       {both_fail} ({100*both_fail/len(common_ids):.1f}%)")
        p(f"  Only {a0['model']:<15} fails: {only_a0_fail} ({100*only_a0_fail/len(common_ids):.1f}%)")
        p(f"  Only {a1['model']:<15} fails: {only_a1_fail} ({100*only_a1_fail/len(common_ids):.1f}%)")
        p(f"  Oracle ensemble: {both_succ + only_a0_fail + only_a1_fail} ({100*(both_succ+only_a0_fail+only_a1_fail)/len(common_ids):.1f}%)")

        # Breakdown of unique failures
        p(f"\n  Why {a0['model']} uniquely fails (SVD correct, Coop wrong):")
        gt_dict_ref = gt_dict_global
        a0_unique_errors = Counter()
        for sid in common_ids:
            if not r0[sid]["success"] and r1[sid]["success"]:
                gt = gt_dict_ref.get(sid)
                if gt is None:
                    continue
                resp = r0[sid].get("response", "")
                fn, args, ps = parse_action_lenient(resp)
                et, _ = classify_error(fn, args, ps, gt)
                if et in ("parse_error", "empty_function", "bad_coordinate_format", "missing_coordinate"):
                    a0_unique_errors["format_error"] += 1
                elif et == "wrong_function":
                    a0_unique_errors["wrong_function"] += 1
                elif et.startswith("wrong_coordinate"):
                    a0_unique_errors["wrong_coordinate"] += 1
                else:
                    a0_unique_errors["wrong_args"] += 1
        for et, c in a0_unique_errors.most_common():
            p(f"    {et}: {c}")

        p(f"\n  Why {a1['model']} uniquely fails (Coop correct, SVD wrong):")
        a1_unique_errors = Counter()
        for sid in common_ids:
            if r0[sid]["success"] and not r1[sid]["success"]:
                gt = gt_dict_ref.get(sid)
                if gt is None:
                    continue
                resp = r1[sid].get("response", "")
                fn, args, ps = parse_action_lenient(resp)
                et, _ = classify_error(fn, args, ps, gt)
                if et in ("parse_error", "empty_function", "bad_coordinate_format", "missing_coordinate"):
                    a1_unique_errors["format_error"] += 1
                elif et == "wrong_function":
                    a1_unique_errors["wrong_function"] += 1
                elif et.startswith("wrong_coordinate"):
                    a1_unique_errors["wrong_coordinate"] += 1
                else:
                    a1_unique_errors["wrong_args"] += 1
        for et, c in a1_unique_errors.most_common():
            p(f"    {et}: {c}")

    # ── Save ──
    report_path = os.path.join(output_dir, "analysis_report_v2.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {report_path}")

    for a in analyses:
        detail_path = os.path.join(output_dir, f"wrong_coord_details_{a['model']}.json")
        with open(detail_path, "w") as f:
            json.dump(a["wrong_coord_details"], f, indent=2, ensure_ascii=False)
        print(f"  Coord details: {detail_path} ({len(a['wrong_coord_details'])} records)")

    # Summary JSON
    summary = []
    for a in analyses:
        summary.append({
            "model": a["model"],
            "total": a["total"],
            "success": a["success"],
            "success_rate": 100 * a["success"] / a["total"] if a["total"] > 0 else 0,
            "error_types_coarse": dict(a["error_types_coarse"]),
            "error_types_fine": dict(a["error_types"]),
            "distance_stats": {
                "count": len(a["distances"]),
                "mean": float(np.mean(a["distances"])) if a["distances"] else 0,
                "median": float(np.median(a["distances"])) if a["distances"] else 0,
            },
            "success_by_region": dict(a["success_by_region"]),
            "total_by_region": dict(a["total_by_region"]),
            "success_by_function": dict(a["success_by_function"]),
            "total_by_function": dict(a["total_by_function"]),
            "success_by_domain": dict(a["success_by_domain"]),
            "total_by_domain": dict(a["total_by_domain"]),
        })
    with open(os.path.join(output_dir, "analysis_summary_v2.json"), "w") as f:
        json.dump(summary, f, indent=2)


gt_dict_global = {}


def main():
    global gt_dict_global

    parser = argparse.ArgumentParser(description="Error analysis v2 (truncation-corrected)")
    parser.add_argument("--test_data_root", required=True)
    parser.add_argument("--result_dirs", nargs="+", required=True,
                        help="model_name=result_dir pairs")
    parser.add_argument("--output_dir", default="evaluation/analysis_results_v2")
    args = parser.parse_args()

    model_dirs = {}
    for rd in args.result_dirs:
        if "=" in rd:
            name, path = rd.split("=", 1)
        else:
            name = os.path.basename(rd.rstrip("/"))
            path = rd
        model_dirs[name] = path

    print(f"Models: {list(model_dirs.keys())}")
    gt_dict_global = load_ground_truth(args.test_data_root)

    analyses = []
    all_results = []
    for name, rdir in model_dirs.items():
        print(f"\nAnalyzing {name}...")
        results = load_results(rdir)
        all_results.append(results)
        analysis = analyze_model(name, results, gt_dict_global)
        analyses.append(analysis)

    print_report(analyses, args.output_dir, all_results)


if __name__ == "__main__":
    main()
