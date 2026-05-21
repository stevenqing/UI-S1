#!/usr/bin/env python3
"""
Per-sample differential analysis: exactly which samples differ, and why.

Outputs:
1. Full sample-level JSONL with both responses + GT for all 19046 samples
2. Detailed breakdown of coop-unique-fail and svd-unique-fail samples
3. Systematic pattern discovery (what distinguishes the differential samples)

Usage:
  python evaluation/analysis_differential_v3.py \
      --test_data_root datasets/GUI-360/test \
      --coop_dir train_GUI_360/GUI-360-eval/results/cooperative_thought_v3_ep2/action_prediction \
      --svd_dir train_GUI_360/GUI-360-eval/results/svd_lora_r256_same_pipeline/action_prediction \
      --output_dir evaluation/analysis_results_v3
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np


# ── Ground truth loading (matching eval script exactly) ───────────

def load_ground_truth(test_data_root):
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

                    sid = f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}_{step_info['line_num']}"
                    args = dict(action.get("args", {}))
                    args.pop("x", None)
                    args.pop("y", None)
                    if action.get("coordinate_x") is not None:
                        args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]

                    gt[sid] = {
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
                        "status": step.get("status", ""),
                    }
    print(f"Loaded {len(gt)} ground truth samples")
    return gt


def load_results(result_dir):
    results = {}
    for fname in sorted(os.listdir(result_dir)):
        if fname.startswith("results_") and fname.endswith(".json"):
            with open(os.path.join(result_dir, fname)) as f:
                data = json.load(f)
            for item in data:
                results[item["sample_id"]] = item
    print(f"  Loaded {len(results)} predictions")
    return results


# ── Lenient parser ────────────────────────────────────────────────

def parse_action_lenient(response):
    if not response or not response.strip():
        return None, None, None, "empty"

    # 1. Full tool_call tags
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("function"), data.get("args", {}), data.get("status"), "full"
        except json.JSONDecodeError:
            pass

    # 2. tool_call without close (truncated)
    m = re.search(r'<tool_call>\s*(\{.*)', response, re.DOTALL)
    if m:
        json_str = m.group(1).strip()
        for suffix in ["}", "]}", "]]}", '"]}', '"]]}', '"]}', '"}']:
            try:
                data = json.loads(json_str + suffix)
                return data.get("function"), data.get("args", {}), data.get("status"), "reconstructed"
            except json.JSONDecodeError:
                continue
        # Extract what we can via regex
        fn_m = re.search(r'"function"\s*:\s*"([^"]*)"', json_str)
        coord_m = re.search(r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)', json_str)
        status_m = re.search(r'"status"\s*:\s*"([^"]*)"', json_str)
        if fn_m:
            fn = fn_m.group(1)
            args = {}
            if coord_m:
                args["coordinate"] = [float(coord_m.group(1)), float(coord_m.group(2))]
            for key in ["button", "double", "text", "keys", "clear_current_text"]:
                km = re.search(rf'"{key}"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|\d+)', json_str)
                if km:
                    try:
                        args[key] = json.loads(km.group(1))
                    except json.JSONDecodeError:
                        args[key] = km.group(1).strip('"')
            return fn, args, status_m.group(1) if status_m else None, "regex"

    # 3. Bare JSON
    m = re.search(r'(\{"function".*?\})', response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("function"), data.get("args", {}), data.get("status"), "bare"
        except json.JSONDecodeError:
            pass

    return None, None, None, "unparseable"


def parse_thought(response):
    m = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
    return m.group(1).strip() if m else ""


# ── Coordinate helpers ────────────────────────────────────────────

def coord_in_rect(x, y, rect):
    return (rect.get("left", 0) <= x <= rect.get("right", 0) and
            rect.get("top", 0) <= y <= rect.get("bottom", 0))

def rect_center(rect):
    return ((rect.get("left", 0) + rect.get("right", 0)) / 2,
            (rect.get("top", 0) + rect.get("bottom", 0)) / 2)

def euclidean_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_pred_coord(pred_args):
    if not pred_args or "coordinate" not in pred_args:
        return None, None
    try:
        return float(pred_args["coordinate"][0]), float(pred_args["coordinate"][1])
    except (TypeError, ValueError, IndexError):
        return None, None


# ── Per-sample error classification ───────────────────────────────

def classify_sample_error(pred_fn, pred_args, pred_status, parse_mode, gt):
    """Returns (error_type, detail_dict)"""
    detail = {}

    if parse_mode in ("empty", "unparseable"):
        return "format:unparseable", detail

    if pred_fn is None or pred_fn == "":
        return "format:empty_function", detail

    gt_fn = gt["function"]
    gt_args = gt["args"]
    rect = gt["rectangle"]

    # Wrong function
    if pred_fn != gt_fn:
        detail["pred_fn"] = pred_fn
        detail["gt_fn"] = gt_fn
        return f"wrong_function:{gt_fn}->{pred_fn}", detail

    # Function correct - check coordinate
    if "coordinate" in gt_args:
        px, py = get_pred_coord(pred_args)
        if px is None:
            return "format:missing_coordinate", detail

        gt_cx, gt_cy = rect_center(rect)
        dist = euclidean_dist(px, py, gt_cx, gt_cy)
        detail["pred_xy"] = [px, py]
        detail["gt_center"] = [gt_cx, gt_cy]
        detail["distance"] = round(dist, 1)
        detail["rect"] = rect

        in_rect = coord_in_rect(px, py, rect)
        if not in_rect:
            # Classify distance
            if dist < 50:
                dist_cat = "near"
            elif dist < 200:
                dist_cat = "medium"
            else:
                dist_cat = "far"
            return f"wrong_coordinate:{dist_cat}({dist:.0f}px)", detail

        # Coordinate correct - check other args
        for key in gt_args:
            if key == "coordinate":
                continue
            if key not in pred_args:
                return f"wrong_args:missing_{key}", detail
            if str(pred_args[key]).lower() != str(gt_args[key]).lower():
                detail["key"] = key
                detail["pred_val"] = str(pred_args[key])
                detail["gt_val"] = str(gt_args[key])
                return f"wrong_args:{key}_mismatch", detail

        return "unknown:all_match_but_failed", detail
    else:
        # No coordinate - check args
        for key in gt_args:
            if not pred_args or key not in pred_args:
                return f"wrong_args:missing_{key}", detail
            if str(pred_args[key]).lower() != str(gt_args[key]).lower():
                return f"wrong_args:{key}_mismatch", detail
        return "unknown:all_match_but_failed", detail


# ── Main analysis ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_root", required=True)
    parser.add_argument("--coop_dir", required=True)
    parser.add_argument("--svd_dir", required=True)
    parser.add_argument("--output_dir", default="evaluation/analysis_results_v3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ground truth...")
    gt = load_ground_truth(args.test_data_root)

    print("Loading coop results...")
    coop = load_results(args.coop_dir)
    print("Loading SVD results...")
    svd = load_results(args.svd_dir)

    common_ids = sorted(set(coop.keys()) & set(svd.keys()) & set(gt.keys()))
    print(f"Common samples: {len(common_ids)}")

    # ── Build per-sample records ──
    records = []
    for sid in common_ids:
        g = gt[sid]
        c = coop[sid]
        s = svd[sid]

        c_resp = c.get("response", "")
        s_resp = s.get("response", "")

        c_fn, c_args, c_status, c_parse = parse_action_lenient(c_resp)
        s_fn, s_args, s_status, s_parse = parse_action_lenient(s_resp)

        c_px, c_py = get_pred_coord(c_args)
        s_px, s_py = get_pred_coord(s_args)
        gt_cx, gt_cy = rect_center(g["rectangle"])

        c_dist = euclidean_dist(c_px, c_py, gt_cx, gt_cy) if c_px is not None else None
        s_dist = euclidean_dist(s_px, s_py, gt_cx, gt_cy) if s_px is not None else None

        # Classify errors for failures
        c_error = None
        s_error = None
        c_detail = {}
        s_detail = {}
        if not c["success"]:
            c_error, c_detail = classify_sample_error(c_fn, c_args, c_status, c_parse, g)
        if not s["success"]:
            s_error, s_detail = classify_sample_error(s_fn, s_args, s_status, s_parse, g)

        # Quadrant
        if c["success"] and s["success"]:
            quadrant = "both_succeed"
        elif not c["success"] and not s["success"]:
            quadrant = "both_fail"
        elif not c["success"] and s["success"]:
            quadrant = "coop_only_fail"
        else:
            quadrant = "svd_only_fail"

        records.append({
            "sample_id": sid,
            "quadrant": quadrant,
            "domain": g["domain"],
            "category": g["category"],
            "step_id": g["step_id"],
            "total_steps": g["total_steps"],
            "gt_function": g["function"],
            "gt_rect": g["rectangle"],
            "gt_coord": [g["gt_x"], g["gt_y"]] if g["gt_x"] is not None else None,
            "gt_thought": g["thought_gt"][:200],
            "request": g["request"][:200],
            "control_text": g["control_text"],
            # Coop
            "coop_success": c["success"],
            "coop_fn": c_fn,
            "coop_coord": [c_px, c_py] if c_px is not None else None,
            "coop_dist": round(c_dist, 1) if c_dist is not None else None,
            "coop_thought": parse_thought(c_resp)[:200],
            "coop_error": c_error,
            "coop_parse_mode": c_parse,
            # SVD
            "svd_success": s["success"],
            "svd_fn": s_fn,
            "svd_coord": [s_px, s_py] if s_px is not None else None,
            "svd_dist": round(s_dist, 1) if s_dist is not None else None,
            "svd_error": s_error,
            "svd_parse_mode": s_parse,
        })

    # ── Save full dataset ──
    with open(os.path.join(args.output_dir, "all_samples.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to all_samples.jsonl")

    # ── Quadrant counts ──
    quadrants = Counter(r["quadrant"] for r in records)
    print(f"\n{'='*70}")
    print("QUADRANT SUMMARY")
    print(f"{'='*70}")
    for q in ["both_succeed", "both_fail", "coop_only_fail", "svd_only_fail"]:
        print(f"  {q:<20}: {quadrants[q]:>5} ({100*quadrants[q]/len(records):.1f}%)")

    # ── Analyze coop-only-fail ──
    coop_only = [r for r in records if r["quadrant"] == "coop_only_fail"]
    svd_only = [r for r in records if r["quadrant"] == "svd_only_fail"]
    both_fail = [r for r in records if r["quadrant"] == "both_fail"]

    print(f"\n{'='*70}")
    print(f"COOP-ONLY FAILURES (N={len(coop_only)}): SVD correct, Coop wrong")
    print(f"{'='*70}")
    _analyze_group(coop_only, "coop", args.output_dir)

    print(f"\n{'='*70}")
    print(f"SVD-ONLY FAILURES (N={len(svd_only)}): Coop correct, SVD wrong")
    print(f"{'='*70}")
    _analyze_group(svd_only, "svd", args.output_dir)

    print(f"\n{'='*70}")
    print(f"BOTH-FAIL ANALYSIS (N={len(both_fail)})")
    print(f"{'='*70}")
    _analyze_both_fail(both_fail, args.output_dir)

    # ── Save differential details ──
    with open(os.path.join(args.output_dir, "coop_only_fail.jsonl"), "w") as f:
        for r in coop_only:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "svd_only_fail.jsonl"), "w") as f:
        for r in svd_only:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved coop_only_fail.jsonl ({len(coop_only)}) and svd_only_fail.jsonl ({len(svd_only)})")


def _analyze_group(records, failing_model, output_dir):
    """Analyze a group of differential failures."""
    prefix = "coop" if failing_model == "coop" else "svd"
    error_key = f"{prefix}_error"
    fn_key = f"{prefix}_fn"
    coord_key = f"{prefix}_coord"
    dist_key = f"{prefix}_dist"

    # 1. Error type distribution
    error_counter = Counter(r[error_key] for r in records if r[error_key])
    print(f"\n  Error type distribution:")
    for err, cnt in error_counter.most_common(20):
        print(f"    {err:<45}: {cnt:>5} ({100*cnt/len(records):.1f}%)")

    # 2. Coarse breakdown
    coarse = Counter()
    for r in records:
        err = r[error_key] or "unknown"
        if err.startswith("wrong_coordinate"):
            coarse["wrong_coordinate"] += 1
        elif err.startswith("wrong_function"):
            coarse["wrong_function"] += 1
        elif err.startswith("wrong_args"):
            coarse["wrong_args"] += 1
        elif err.startswith("format"):
            coarse["format_error"] += 1
        else:
            coarse["other"] += 1
    print(f"\n  Coarse breakdown:")
    for c, cnt in coarse.most_common():
        print(f"    {c:<25}: {cnt:>5} ({100*cnt/len(records):.1f}%)")

    # 3. By GT function
    fn_counter = Counter(r["gt_function"] for r in records)
    print(f"\n  By GT function (where {failing_model} uniquely fails):")
    for fn, cnt in fn_counter.most_common(10):
        total_fn = sum(1 for r in records if r["gt_function"] == fn)
        # What error types for this function?
        fn_errors = Counter()
        for r in records:
            if r["gt_function"] == fn:
                err = r[error_key] or "unknown"
                if err.startswith("wrong_coordinate"):
                    fn_errors["wrong_coord"] += 1
                elif err.startswith("wrong_function"):
                    fn_errors["wrong_fn"] += 1
                elif err.startswith("wrong_args"):
                    fn_errors["wrong_args"] += 1
                else:
                    fn_errors["other"] += 1
        err_str = ", ".join(f"{k}={v}" for k, v in fn_errors.most_common())
        print(f"    {fn:<22}: {cnt:>5}  ({err_str})")

    # 4. Function confusion detail (for wrong_function errors)
    fn_confusion = Counter()
    for r in records:
        err = r[error_key] or ""
        if err.startswith("wrong_function:"):
            confusion = err.split(":", 1)[1]
            fn_confusion[confusion] += 1
    if fn_confusion:
        print(f"\n  Function confusions (GT->Pred):")
        for conf, cnt in fn_confusion.most_common(15):
            print(f"    {conf:<35}: {cnt:>5}")

    # 5. Distance distribution for wrong_coordinate
    wc_records = [r for r in records if (r[error_key] or "").startswith("wrong_coordinate")]
    if wc_records:
        dists = [r[dist_key] for r in wc_records if r[dist_key] is not None]
        if dists:
            arr = np.array(dists)
            print(f"\n  Wrong coordinate distances (N={len(dists)}):")
            print(f"    mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
                  f"p25={np.percentile(arr,25):.1f}, p75={np.percentile(arr,75):.1f}")
            bins = [(0, 50, "near"), (50, 200, "medium"), (200, 500, "far"), (500, 9999, "very_far")]
            for lo, hi, label in bins:
                cnt = sum(1 for d in dists if lo <= d < hi)
                print(f"    {label} ({lo}-{hi}px): {cnt} ({100*cnt/len(dists):.1f}%)")

    # 6. By domain
    domain_counter = Counter(r["domain"] for r in records)
    print(f"\n  By domain:")
    for dom, cnt in domain_counter.most_common():
        print(f"    {dom}: {cnt}")

    # 7. By step position
    def step_pos(r):
        if r["total_steps"] > 0:
            ratio = r["step_id"] / r["total_steps"]
            return "early" if ratio <= 0.33 else ("mid" if ratio <= 0.66 else "late")
        return "unknown"
    pos_counter = Counter(step_pos(r) for r in records)
    print(f"\n  By step position:")
    for pos in ["early", "mid", "late"]:
        print(f"    {pos}: {pos_counter.get(pos, 0)}")

    # 8. Sample examples (top error types)
    lines = []
    for err_type, _ in error_counter.most_common(5):
        examples = [r for r in records if r[error_key] == err_type][:3]
        lines.append(f"\n### {err_type} (N={error_counter[err_type]})")
        for r in examples:
            lines.append(f"\n  {r['sample_id']}")
            lines.append(f"  Request: {r['request'][:150]}")
            lines.append(f"  GT: {r['gt_function']}  coord={r['gt_coord']}  rect={r['gt_rect']}")
            lines.append(f"  Coop: fn={r['coop_fn']}  coord={r['coop_coord']}  dist={r['coop_dist']}  err={r['coop_error']}")
            lines.append(f"  SVD:  fn={r['svd_fn']}  coord={r['svd_coord']}  dist={r['svd_dist']}  err={r['svd_error']}")
            if r.get("coop_thought"):
                lines.append(f"  Coop thought: {r['coop_thought'][:150]}")
            lines.append(f"  GT thought: {r['gt_thought'][:150]}")

    with open(os.path.join(output_dir, f"{prefix}_only_fail_examples.txt"), "w") as f:
        f.write("\n".join(lines))


def _analyze_both_fail(records, output_dir):
    """Analyze samples where both models fail — who is closer?"""

    # Compare distances when both have wrong coordinates
    both_coord = []
    for r in records:
        c_err = r["coop_error"] or ""
        s_err = r["svd_error"] or ""
        if c_err.startswith("wrong_coordinate") and s_err.startswith("wrong_coordinate"):
            if r["coop_dist"] is not None and r["svd_dist"] is not None:
                both_coord.append(r)

    print(f"\n  Both wrong_coordinate (N={len(both_coord)}):")
    if both_coord:
        coop_closer = sum(1 for r in both_coord if r["coop_dist"] < r["svd_dist"])
        svd_closer = sum(1 for r in both_coord if r["svd_dist"] < r["coop_dist"])
        tied = len(both_coord) - coop_closer - svd_closer
        print(f"    Coop closer to target: {coop_closer} ({100*coop_closer/len(both_coord):.1f}%)")
        print(f"    SVD closer to target:  {svd_closer} ({100*svd_closer/len(both_coord):.1f}%)")
        print(f"    Tied:                  {tied}")

        c_dists = np.array([r["coop_dist"] for r in both_coord])
        s_dists = np.array([r["svd_dist"] for r in both_coord])
        print(f"    Coop mean dist: {c_dists.mean():.1f}, median: {np.median(c_dists):.1f}")
        print(f"    SVD  mean dist: {s_dists.mean():.1f}, median: {np.median(s_dists):.1f}")

    # Same error vs different error
    same_error_type = 0
    diff_error_type = 0
    for r in records:
        c_coarse = _coarse(r["coop_error"])
        s_coarse = _coarse(r["svd_error"])
        if c_coarse == s_coarse:
            same_error_type += 1
        else:
            diff_error_type += 1

    print(f"\n  Error type agreement:")
    print(f"    Same coarse error: {same_error_type} ({100*same_error_type/len(records):.1f}%)")
    print(f"    Different error:   {diff_error_type} ({100*diff_error_type/len(records):.1f}%)")

    # When errors differ, what are the patterns?
    diff_patterns = Counter()
    for r in records:
        c_coarse = _coarse(r["coop_error"])
        s_coarse = _coarse(r["svd_error"])
        if c_coarse != s_coarse:
            diff_patterns[f"coop:{c_coarse} / svd:{s_coarse}"] += 1
    print(f"\n  Different-error patterns:")
    for pat, cnt in diff_patterns.most_common(10):
        print(f"    {pat}: {cnt}")

    # Both wrong_function — do they predict the SAME wrong function?
    both_wf = [r for r in records
                if (r["coop_error"] or "").startswith("wrong_function")
                and (r["svd_error"] or "").startswith("wrong_function")]
    if both_wf:
        same_pred = sum(1 for r in both_wf if r["coop_fn"] == r["svd_fn"])
        print(f"\n  Both wrong_function (N={len(both_wf)}):")
        print(f"    Same wrong prediction: {same_pred} ({100*same_pred/len(both_wf):.1f}%)")
        print(f"    Different prediction:  {len(both_wf)-same_pred}")


def _coarse(error):
    if error is None:
        return "none"
    if error.startswith("wrong_coordinate"):
        return "wrong_coord"
    if error.startswith("wrong_function"):
        return "wrong_fn"
    if error.startswith("wrong_args") or error.startswith("missing_arg"):
        return "wrong_args"
    if error.startswith("format"):
        return "format"
    return "other"


if __name__ == "__main__":
    main()
