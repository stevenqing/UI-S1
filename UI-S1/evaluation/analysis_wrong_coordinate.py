#!/usr/bin/env python3
"""
Detailed error analysis for action prediction: cooperative LoRA vs SVD LoRA.

Dimensions:
1. Error type breakdown (wrong_function, wrong_coordinate, parse_error, wrong_args)
2. Error distance distribution for wrong_coordinate (near-miss vs far-miss)
3. UI region analysis (ribbon/toolbar/worksheet/dialog/statusbar)
4. Function type breakdown (click, type, scroll, etc.)
5. Step index distribution (early vs mid vs late steps)

Usage:
  python evaluation/analysis_wrong_coordinate.py \
      --test_data_root datasets/GUI-360/test \
      --result_dirs cooperative_thought_v3_ep2=train_GUI_360/GUI-360-eval/results/cooperative_thought_v3_ep2/action_prediction \
                     svd_lora_r256=train_GUI_360/GUI-360-eval/results/svd_lora_r256_same_pipeline/action_prediction \
      --output_dir evaluation/analysis_results
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
    """Load all test samples with ground truth actions.

    Mirrors the eval script's loading logic exactly:
    - Only from success/ subdirectory
    - Only samples with "action_prediction" tag
    - Skips drag actions and missing rectangles
    - Transforms args: removes x/y, adds coordinate=[coord_x, coord_y]

    Returns dict: sample_id -> {action, rectangle, domain, step_id, total_steps, ...}
    """
    data_path = os.path.join(test_data_root, "data")
    gt = {}
    line_count = 0

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
                with open(fpath) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        step = data.get("step", {})
                        tags = step.get("tags", [])

                        # Match eval: only action_prediction samples
                        if "action_prediction" not in tags:
                            continue

                        action = step.get("action", {})

                        # Match eval: skip drag and missing rectangles
                        if action.get("function") == "drag" or not action.get("rectangle"):
                            continue

                        sample_id = f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}_{line_num}"

                        # Transform args to match eval format
                        args = dict(action.get("args", {}))
                        args.pop("x", None)
                        args.pop("y", None)
                        if action.get("coordinate_x") is not None:
                            args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]

                        gt[sample_id] = {
                            "domain": domain,
                            "category": category,
                            "step_id": data.get("step_id", line_num),
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
                        line_count += 1

    print(f"Loaded {line_count} ground truth samples")
    return gt


# ── Load prediction results ───────────────────────────────────────

def load_results(result_dir):
    """Load all shard results from a directory. Returns list of {sample_id, success, response}."""
    results = []
    for fname in sorted(os.listdir(result_dir)):
        if fname.startswith("results_") and fname.endswith(".json"):
            with open(os.path.join(result_dir, fname)) as f:
                data = json.load(f)
            results.extend(data)
    print(f"  Loaded {len(results)} predictions from {result_dir}")
    return results


# ── Parsing helpers ───────────────────────────────────────────────

def parse_action(response):
    """Parse function, args, status from model response."""
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


def parse_thought(response):
    """Extract thought text from <thought>...</thought> tags."""
    m = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
    return m.group(1).strip() if m else ""


def coord_in_rect(x, y, rect):
    """Check if (x,y) falls within rectangle."""
    return (rect.get("left", 0) <= x <= rect.get("right", 0) and
            rect.get("top", 0) <= y <= rect.get("bottom", 0))


def rect_center(rect):
    """Get center of rectangle."""
    cx = (rect.get("left", 0) + rect.get("right", 0)) / 2
    cy = (rect.get("top", 0) + rect.get("bottom", 0)) / 2
    return cx, cy


def euclidean_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ── Error classification ─────────────────────────────────────────

def classify_error(pred_fn, pred_args, parsed_ok, gt_info):
    """Classify an error into detailed categories.

    Returns: error_type, extra_info dict
    """
    extra = {}

    if not parsed_ok:
        return "parse_error", extra

    if pred_fn is None:
        return "parse_error", extra

    gt_fn = gt_info["function"]
    gt_args = gt_info["args"]
    rect = gt_info["rectangle"]

    if pred_fn != gt_fn:
        extra["pred_function"] = pred_fn
        extra["gt_function"] = gt_fn
        return "wrong_function", extra

    # Function matches — check args
    if "coordinate" in gt_args and pred_args and "coordinate" in pred_args:
        try:
            px, py = float(pred_args["coordinate"][0]), float(pred_args["coordinate"][1])
        except (TypeError, ValueError, IndexError):
            return "parse_error", extra

        in_rect = coord_in_rect(px, py, rect)
        if not in_rect:
            # Wrong coordinate — compute distance to GT center
            gt_cx, gt_cy = rect_center(rect)
            dist = euclidean_dist(px, py, gt_cx, gt_cy)
            extra["pred_x"] = px
            extra["pred_y"] = py
            extra["gt_cx"] = gt_cx
            extra["gt_cy"] = gt_cy
            extra["distance"] = dist
            extra["rect"] = rect
            return "wrong_coordinate", extra

        # Coordinate OK — check other args
        for key in gt_args:
            if key == "coordinate":
                continue
            if key not in pred_args:
                extra["missing_key"] = key
                return "wrong_args", extra
            if str(pred_args[key]).lower() != str(gt_args[key]).lower():
                extra["key"] = key
                extra["pred_val"] = str(pred_args[key])
                extra["gt_val"] = str(gt_args[key])
                return "wrong_args", extra

        # Everything should match — this shouldn't happen since success=False
        return "unknown_error", extra

    elif pred_args and gt_args:
        # No coordinate — check other args
        for key in gt_args:
            if key not in pred_args:
                extra["missing_key"] = key
                return "wrong_args", extra
            if str(pred_args[key]).lower() != str(gt_args[key]).lower():
                extra["key"] = key
                return "wrong_args", extra

    return "unknown_error", extra


# ── UI region classification ──────────────────────────────────────

def classify_ui_region(gt_y, gt_x=None, domain="excel"):
    """Classify target UI region based on GT coordinate position.

    Rough heuristics for Office apps at ~1024x768 resolution:
    - Title bar: y < 30
    - Ribbon/toolbar: 30 <= y < 130
    - Formula bar (Excel): 130 <= y < 160
    - Worksheet/content area: 160 <= y < 700
    - Status bar: y >= 700
    """
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


# ── Analysis functions ────────────────────────────────────────────

def analyze_model(model_name, results, gt_dict):
    """Run full analysis on one model's predictions."""
    analysis = {
        "model": model_name,
        "total": 0,
        "success": 0,
        "fail": 0,
        "error_types": Counter(),
        "distances": [],  # for wrong_coordinate only
        "errors_by_region": defaultdict(Counter),  # region -> error_type -> count
        "errors_by_function": defaultdict(Counter),  # gt_function -> error_type -> count
        "errors_by_step_pos": defaultdict(Counter),  # step_position -> error_type -> count
        "success_by_region": Counter(),
        "total_by_region": Counter(),
        "success_by_function": Counter(),
        "total_by_function": Counter(),
        "success_by_step_pos": Counter(),
        "total_by_step_pos": Counter(),
        "wrong_coord_details": [],  # list of detailed wrong_coordinate records
    }

    for item in results:
        sid = item["sample_id"]
        gt = gt_dict.get(sid)
        if gt is None:
            continue

        analysis["total"] += 1

        # Classify step position
        step_id = gt["step_id"]
        total_steps = gt["total_steps"]
        if total_steps > 0:
            ratio = step_id / total_steps
            if ratio <= 0.33:
                step_pos = "early"
            elif ratio <= 0.66:
                step_pos = "mid"
            else:
                step_pos = "late"
        else:
            step_pos = "unknown"

        # UI region
        region = classify_ui_region(gt["gt_y"], gt["gt_x"], gt["domain"])
        gt_fn = gt["function"]

        analysis["total_by_region"][region] += 1
        analysis["total_by_function"][gt_fn] += 1
        analysis["total_by_step_pos"][step_pos] += 1

        if item["success"]:
            analysis["success"] += 1
            analysis["success_by_region"][region] += 1
            analysis["success_by_function"][gt_fn] += 1
            analysis["success_by_step_pos"][step_pos] += 1
            continue

        analysis["fail"] += 1

        # Classify error
        response = item.get("response", "")
        pred_fn, pred_args, parsed_ok = parse_action(response)
        error_type, extra = classify_error(pred_fn, pred_args, parsed_ok, gt)

        analysis["error_types"][error_type] += 1
        analysis["errors_by_region"][region][error_type] += 1
        analysis["errors_by_function"][gt_fn][error_type] += 1
        analysis["errors_by_step_pos"][step_pos][error_type] += 1

        if error_type == "wrong_coordinate":
            analysis["distances"].append(extra["distance"])
            # Store detailed record
            thought_pred = parse_thought(response)
            analysis["wrong_coord_details"].append({
                "sample_id": sid,
                "domain": gt["domain"],
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


def print_report(analyses, output_dir):
    """Print and save comparative report."""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    def row(*parts):
        """Build a row from parts, print and save."""
        line = "".join(parts)
        print(line)
        lines.append(line)

    p("=" * 80)
    p("ERROR ANALYSIS: Cooperative LoRA vs SVD LoRA")
    p("=" * 80)

    # ── 1. Overall + Error Type Breakdown ──
    p("\n## 1. Overall Results & Error Type Breakdown\n")
    header = f"{'Model':<30} {'Total':>6} {'Success':>8} {'Rate':>7}"
    p(header)
    p("-" * len(header))
    for a in analyses:
        rate = 100 * a["success"] / a["total"] if a["total"] > 0 else 0
        p(f"{a['model']:<30} {a['total']:>6} {a['success']:>8} {rate:>6.1f}%")

    parts = [f"\n{'Error Type':<25}"]
    for a in analyses:
        parts.append(f" {a['model']:>20}")
    row(*parts)
    p("-" * (25 + 21 * len(analyses)))

    all_error_types = sorted(set().union(*(a["error_types"].keys() for a in analyses)))
    for et in all_error_types:
        parts = [f"{et:<25}"]
        for a in analyses:
            count = a["error_types"].get(et, 0)
            pct = 100 * count / a["fail"] if a["fail"] > 0 else 0
            parts.append(f" {count:>8} ({pct:>5.1f}%)")
        row(*parts)

    # ── 2. Distance Distribution (wrong_coordinate) ──
    p("\n## 2. Wrong Coordinate Distance Distribution\n")
    dist_bins = [0, 20, 50, 100, 200, 500, float("inf")]
    bin_labels = ["0-20 (near-miss)", "20-50", "50-100", "100-200", "200-500", "500+ (far-miss)"]

    parts = [f"{'Distance bin':<25}"]
    for a in analyses:
        parts.append(f" {a['model']:>20}")
    row(*parts)
    p("-" * (25 + 21 * len(analyses)))

    for i, label in enumerate(bin_labels):
        parts = [f"{label:<25}"]
        for a in analyses:
            dists = a["distances"]
            count = sum(1 for d in dists if dist_bins[i] <= d < dist_bins[i+1])
            pct = 100 * count / len(dists) if dists else 0
            parts.append(f" {count:>8} ({pct:>5.1f}%)")
        row(*parts)

    for a in analyses:
        dists = a["distances"]
        if dists:
            arr = np.array(dists)
            p(f"\n  {a['model']}: mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
              f"p25={np.percentile(arr,25):.1f}, p75={np.percentile(arr,75):.1f}, "
              f"p90={np.percentile(arr,90):.1f}, max={arr.max():.1f}")

    # ── 3. UI Region Analysis ──
    p("\n## 3. Success Rate by UI Region\n")
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

    # Error type breakdown per region
    p("\n  Error types per region:")
    for region in all_regions:
        for a in analyses:
            total_fail_region = sum(a["errors_by_region"].get(region, {}).values())
            if total_fail_region > 10:
                et_parts = []
                for et in all_error_types:
                    c = a["errors_by_region"].get(region, {}).get(et, 0)
                    if c > 0:
                        et_parts.append(f"{et}={c}")
                p(f"    {a['model']} | {region}: {' '.join(et_parts)}")

    # ── 4. Function Type Breakdown ──
    p("\n## 4. Success Rate by Function Type\n")
    all_fns = sorted(set().union(*(a["total_by_function"].keys() for a in analyses)))

    parts = [f"{'Function':<20}"]
    for a in analyses:
        parts.append(f" {a['model']:>25}")
    row(*parts)
    p("-" * (20 + 26 * len(analyses)))

    for fn in all_fns:
        parts = [f"{fn:<20}"]
        for a in analyses:
            total = a["total_by_function"].get(fn, 0)
            succ = a["success_by_function"].get(fn, 0)
            rate = 100 * succ / total if total > 0 else 0
            parts.append(f" {succ:>5}/{total:<5} ({rate:>5.1f}%)")
        row(*parts)

    # ── 5. Step Position Analysis ──
    p("\n## 5. Success Rate by Step Position\n")
    all_pos = ["early", "mid", "late", "unknown"]

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

    # ── 6. Comparative: where does each model uniquely fail? ──
    if len(analyses) == 2:
        p("\n## 6. Differential Analysis (A fails, B succeeds & vice versa)\n")
        a0, a1 = analyses[0], analyses[1]
        r0 = {item["sample_id"]: item["success"] for item in load_results_raw[0]}
        r1 = {item["sample_id"]: item["success"] for item in load_results_raw[1]}

        common_ids = set(r0.keys()) & set(r1.keys())
        only_a0_fail = sum(1 for sid in common_ids if not r0[sid] and r1[sid])
        only_a1_fail = sum(1 for sid in common_ids if r0[sid] and not r1[sid])
        both_fail = sum(1 for sid in common_ids if not r0[sid] and not r1[sid])
        both_succ = sum(1 for sid in common_ids if r0[sid] and r1[sid])

        p(f"  Common samples: {len(common_ids)}")
        p(f"  Both succeed:    {both_succ}")
        p(f"  Both fail:       {both_fail}")
        p(f"  Only {a0['model']:<15} fails: {only_a0_fail}")
        p(f"  Only {a1['model']:<15} fails: {only_a1_fail}")

    # Save report
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {report_path}")

    # Save detailed wrong_coordinate records
    for a in analyses:
        detail_path = os.path.join(output_dir, f"wrong_coord_details_{a['model']}.json")
        with open(detail_path, "w") as f:
            json.dump(a["wrong_coord_details"], f, indent=2, ensure_ascii=False)
        print(f"  Details saved: {detail_path} ({len(a['wrong_coord_details'])} records)")

    # Save full analysis as JSON (for downstream use)
    summary = []
    for a in analyses:
        summary.append({
            "model": a["model"],
            "total": a["total"],
            "success": a["success"],
            "success_rate": 100 * a["success"] / a["total"] if a["total"] > 0 else 0,
            "error_types": dict(a["error_types"]),
            "distance_stats": {
                "count": len(a["distances"]),
                "mean": float(np.mean(a["distances"])) if a["distances"] else 0,
                "median": float(np.median(a["distances"])) if a["distances"] else 0,
                "p25": float(np.percentile(a["distances"], 25)) if a["distances"] else 0,
                "p75": float(np.percentile(a["distances"], 75)) if a["distances"] else 0,
            },
            "success_by_region": dict(a["success_by_region"]),
            "total_by_region": dict(a["total_by_region"]),
            "success_by_function": dict(a["success_by_function"]),
            "total_by_function": dict(a["total_by_function"]),
            "success_by_step_pos": dict(a["success_by_step_pos"]),
            "total_by_step_pos": dict(a["total_by_step_pos"]),
        })

    with open(os.path.join(output_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# Global for differential analysis
load_results_raw = []


def main():
    global load_results_raw

    parser = argparse.ArgumentParser(description="Error analysis for action prediction")
    parser.add_argument("--test_data_root", required=True,
                        help="Path to GUI-360 test data root (e.g. datasets/GUI-360/test)")
    parser.add_argument("--result_dirs", nargs="+", required=True,
                        help="model_name=result_dir pairs (e.g. coop_v3=path/to/results)")
    parser.add_argument("--output_dir", default="evaluation/analysis_results",
                        help="Output directory for analysis results")
    args = parser.parse_args()

    # Parse result dir pairs
    model_dirs = {}
    for rd in args.result_dirs:
        if "=" in rd:
            name, path = rd.split("=", 1)
        else:
            name = os.path.basename(rd.rstrip("/"))
            path = rd
        model_dirs[name] = path

    print(f"Models to analyze: {list(model_dirs.keys())}")
    print(f"Loading ground truth from {args.test_data_root}...")
    gt = load_ground_truth(args.test_data_root)

    analyses = []
    for name, rdir in model_dirs.items():
        print(f"\nAnalyzing {name}...")
        results = load_results(rdir)
        load_results_raw.append(results)
        analysis = analyze_model(name, results, gt)
        analyses.append(analysis)

    print_report(analyses, args.output_dir)


if __name__ == "__main__":
    main()
