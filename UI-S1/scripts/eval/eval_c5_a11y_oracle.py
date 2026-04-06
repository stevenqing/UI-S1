#!/usr/bin/env python3
"""
Exp C5: A11y Oracle Analysis

For hard cases (both V2 and V3 wrong), check if a11y element labels
provide disambiguating information that could improve grounding.

Questions:
1. What fraction of hard cases have a distinctive a11y label for the GT element?
2. Among those, does the label name/type clearly describe the target?
3. How many unique element types are involved?
4. Can element_id-based actions bypass coordinate errors entirely?

Data sources:
- outputs/eval_c/hard_case_ids.json (3,326 hard case sample IDs)
- outputs/exp1_1/results_K10.jsonl (grounding results with GT rectangles)
- train_GUI_360/data/gui360_test_a11y_eval.parquet (a11y element data)
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_hard_case_ids():
    path = PROJECT_ROOT / "outputs" / "eval_c" / "hard_case_ids.json"
    with open(path) as f:
        return json.load(f)


def load_k10_results():
    path = PROJECT_ROOT / "outputs" / "exp1_1" / "results_K10.jsonl"
    results = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            results[d["sample_id"]] = d
    return results


def map_sample_id_to_a11y_id(sample_id):
    """Map K=10 sample_id to a11y trajectory+step.

    K=10 ID: 'excel_search_excel_4s_11596_3' → domain=excel, category=search
    a11y ID: trajectory_id='excel_4s_11596', step_id=3
    """
    parts = sample_id.split("_")
    # Format: domain_category_domain_Xs_NNNNN_step
    # e.g., excel_search_excel_4s_11596_3
    # → strip domain_category_ prefix: excel_4s_11596_3
    # → trajectory_id = excel_4s_11596, step_id = 3

    # Find the second occurrence of domain name
    domain = parts[0]
    category = parts[1]
    # Rest after domain_category_
    rest = "_".join(parts[2:])
    # Split off last part as step
    rest_parts = rest.rsplit("_", 1)
    if len(rest_parts) == 2:
        traj_id = rest_parts[0]
        step_id = int(rest_parts[1])
        return traj_id, step_id
    return None, None


def load_a11y_data():
    """Load a11y test data and index by (trajectory_id, step_id)."""
    path = PROJECT_ROOT / "train_GUI_360" / "data" / "gui360_test_a11y_eval.parquet"
    df = pd.read_parquet(path)
    index = {}
    for _, row in df.iterrows():
        key = (row["trajectory_id"], int(row["step_id"]))
        index[key] = row
    return index


def parse_element_list(messages_str):
    """Extract element list from a11y messages."""
    if isinstance(messages_str, str):
        messages = json.loads(messages_str)
    else:
        messages = messages_str

    elements = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text = part["text"]
                    # Parse lines like: [1] MenuItem "qabench_1000036" (position: 196,24)
                    for line in text.split("\n"):
                        m = re.match(
                            r'\[(\d+)\]\s+(\w+)\s+"([^"]*?)"\s+\(position:\s*(\d+),(\d+)\)',
                            line.strip()
                        )
                        if m:
                            elements.append({
                                "id": int(m.group(1)),
                                "type": m.group(2),
                                "text": m.group(3),
                                "x": int(m.group(4)),
                                "y": int(m.group(5)),
                            })
        elif isinstance(content, str):
            for line in content.split("\n"):
                m = re.match(
                    r'\[(\d+)\]\s+(\w+)\s+"([^"]*?)"\s+\(position:\s*(\d+),(\d+)\)',
                    line.strip()
                )
                if m:
                    elements.append({
                        "id": int(m.group(1)),
                        "type": m.group(2),
                        "text": m.group(3),
                        "x": int(m.group(4)),
                        "y": int(m.group(5)),
                    })
    return elements


def find_gt_element(elements, gt_rect):
    """Find the a11y element whose position falls within the GT rectangle."""
    if not gt_rect or not elements:
        return None

    left = gt_rect.get("left", 0)
    top = gt_rect.get("top", 0)
    right = gt_rect.get("right", 0)
    bottom = gt_rect.get("bottom", 0)

    matches = []
    for e in elements:
        if left <= e["x"] <= right and top <= e["y"] <= bottom:
            matches.append(e)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Return the one closest to center
        cx = (left + right) / 2
        cy = (top + bottom) / 2
        matches.sort(key=lambda e: (e["x"] - cx) ** 2 + (e["y"] - cy) ** 2)
        return matches[0]
    return None


def analyze_a11y_coverage(hard_ids, k10_results, a11y_index):
    """Analyze how many hard cases have a11y elements matching GT."""
    print("=" * 70)
    print("  C5.1: A11y Element Coverage for Hard Cases")
    print("=" * 70)

    matched_to_a11y = 0
    has_gt_element = 0
    no_gt_element = 0
    not_in_a11y = 0

    element_types = Counter()
    element_text_lengths = []
    distinctive_labels = 0  # element text is descriptive (not empty/generic)

    gt_element_examples = []

    for sid in hard_ids:
        k10 = k10_results.get(sid)
        if k10 is None:
            continue

        traj_id, step_id = map_sample_id_to_a11y_id(sid)
        if traj_id is None:
            not_in_a11y += 1
            continue

        a11y_row = a11y_index.get((traj_id, step_id))
        if a11y_row is None:
            not_in_a11y += 1
            continue

        matched_to_a11y += 1

        # Parse element list
        elements = parse_element_list(a11y_row["messages"])
        gt_rect = k10.get("gt_rectangle", {})

        gt_elem = find_gt_element(elements, gt_rect)
        if gt_elem:
            has_gt_element += 1
            element_types[gt_elem["type"]] += 1
            element_text_lengths.append(len(gt_elem["text"]))

            # Is the label distinctive?
            text = gt_elem["text"].strip()
            generic = {"", "None", "null", "undefined"}
            if text and text not in generic and not text.startswith("m365_"):
                distinctive_labels += 1

            if len(gt_element_examples) < 10:
                gt_element_examples.append({
                    "sample_id": sid,
                    "element": gt_elem,
                    "n_elements": len(elements),
                })
        else:
            no_gt_element += 1

    total = len(hard_ids)
    print(f"\n  Hard cases total: {total}")
    print(f"  Matched to a11y data: {matched_to_a11y} ({matched_to_a11y/total:.1%})")
    print(f"  Not in a11y data: {not_in_a11y}")
    print(f"\n  Among matched:")
    print(f"    GT element found in a11y: {has_gt_element} ({has_gt_element/matched_to_a11y:.1%})")
    print(f"    GT element NOT found:     {no_gt_element} ({no_gt_element/matched_to_a11y:.1%})")
    print(f"    Distinctive label:        {distinctive_labels} ({distinctive_labels/matched_to_a11y:.1%})")

    print(f"\n  GT element type distribution:")
    for etype, cnt in element_types.most_common(15):
        print(f"    {etype:<20s} {cnt:>5d} ({cnt/has_gt_element:.1%})")

    if element_text_lengths:
        print(f"\n  GT element text length: mean={np.mean(element_text_lengths):.1f}, "
              f"median={np.median(element_text_lengths):.0f}, "
              f"empty={sum(1 for l in element_text_lengths if l == 0)}")

    print(f"\n  Example GT elements for hard cases:")
    for ex in gt_element_examples[:5]:
        e = ex["element"]
        print(f"    {ex['sample_id']}: [{e['id']}] {e['type']} \"{e['text'][:40]}\" "
              f"at ({e['x']},{e['y']}), {ex['n_elements']} total elements")

    return has_gt_element, matched_to_a11y, distinctive_labels


def analyze_element_disambiguation(hard_ids, k10_results, a11y_index):
    """Check if a11y elements could disambiguate between V3 candidates."""
    print(f"\n{'=' * 70}")
    print("  C5.2: A11y Element Disambiguation Potential")
    print("=" * 70)

    # For hard cases with a11y data, check:
    # 1. How many V3 candidates land on different a11y elements?
    # 2. Does any candidate land on the correct a11y element?

    total_analyzed = 0
    candidates_on_different_elements = 0
    any_candidate_on_gt_element = 0

    for sid in hard_ids:
        k10 = k10_results.get(sid)
        if k10 is None:
            continue

        traj_id, step_id = map_sample_id_to_a11y_id(sid)
        if traj_id is None:
            continue

        a11y_row = a11y_index.get((traj_id, step_id))
        if a11y_row is None:
            continue

        elements = parse_element_list(a11y_row["messages"])
        gt_rect = k10.get("gt_rectangle", {})
        all_coords = k10.get("all_coords", [])

        if not elements or len(all_coords) < 2:
            continue

        total_analyzed += 1

        # Find which element each candidate coord falls on
        candidate_elements = set()
        gt_elem = find_gt_element(elements, gt_rect)
        gt_elem_id = gt_elem["id"] if gt_elem else None

        has_gt = False
        for coord in all_coords:
            # Find closest element to this coord
            if not coord or len(coord) < 2:
                continue
            x, y = float(coord[0]), float(coord[1])
            best_elem = None
            best_dist = float("inf")
            for e in elements:
                d = (e["x"] - x) ** 2 + (e["y"] - y) ** 2
                if d < best_dist:
                    best_dist = d
                    best_elem = e

            if best_elem:
                candidate_elements.add(best_elem["id"])
                if best_elem["id"] == gt_elem_id:
                    has_gt = True

        if len(candidate_elements) > 1:
            candidates_on_different_elements += 1
        if has_gt:
            any_candidate_on_gt_element += 1

    print(f"\n  Analyzed: {total_analyzed} hard cases with a11y + multi-candidates")
    print(f"  Candidates land on different elements: {candidates_on_different_elements} "
          f"({candidates_on_different_elements/total_analyzed:.1%})")
    print(f"  At least one candidate near GT element: {any_candidate_on_gt_element} "
          f"({any_candidate_on_gt_element/total_analyzed:.1%})")
    print(f"\n  → If candidates are on different elements, a11y-aware selector could choose correctly")
    print(f"  → Oracle a11y selector ceiling: {any_candidate_on_gt_element/total_analyzed:.1%}")


def analyze_element_id_bypass(hard_ids, k10_results, a11y_index):
    """Check if element_id actions could bypass coordinate errors entirely."""
    print(f"\n{'=' * 70}")
    print("  C5.3: Element-ID Bypass Potential")
    print("=" * 70)

    # In a11y mode, actions use element_id instead of coordinates
    # If the model correctly identifies the element_id, coordinates don't matter

    total = 0
    has_single_match = 0  # GT rect matches exactly one element
    has_multiple_match = 0
    has_no_match = 0

    # Check how often element types are unique enough for selection
    unique_type_text = 0

    for sid in hard_ids:
        k10 = k10_results.get(sid)
        if k10 is None:
            continue

        traj_id, step_id = map_sample_id_to_a11y_id(sid)
        if traj_id is None:
            continue

        a11y_row = a11y_index.get((traj_id, step_id))
        if a11y_row is None:
            continue

        elements = parse_element_list(a11y_row["messages"])
        gt_rect = k10.get("gt_rectangle", {})

        if not elements:
            continue

        total += 1

        left = gt_rect.get("left", 0)
        top = gt_rect.get("top", 0)
        right = gt_rect.get("right", 0)
        bottom = gt_rect.get("bottom", 0)

        matches = [e for e in elements
                   if left <= e["x"] <= right and top <= e["y"] <= bottom]

        if len(matches) == 1:
            has_single_match += 1
            # Is this element's (type, text) unique in the element list?
            elem = matches[0]
            same_type_text = sum(1 for e in elements
                                if e["type"] == elem["type"] and e["text"] == elem["text"])
            if same_type_text == 1:
                unique_type_text += 1
        elif len(matches) > 1:
            has_multiple_match += 1
        else:
            has_no_match += 1

    print(f"\n  Hard cases with a11y: {total}")
    print(f"  GT matches exactly 1 element: {has_single_match} ({has_single_match/total:.1%})")
    print(f"  GT matches >1 elements:       {has_multiple_match} ({has_multiple_match/total:.1%})")
    print(f"  GT matches no elements:       {has_no_match} ({has_no_match/total:.1%})")
    print(f"\n  Among single-match:")
    print(f"    Unique (type,text) in screen: {unique_type_text} ({unique_type_text/has_single_match:.1%})")
    print(f"  → These could be addressed by element_id prediction instead of coordinate")


def main():
    print("Loading hard case IDs...")
    hard_ids = load_hard_case_ids()
    print(f"Loaded {len(hard_ids)} hard case IDs")

    print("Loading K=10 results...")
    k10_results = load_k10_results()
    print(f"Loaded {len(k10_results)} K=10 results")

    print("Loading a11y data...")
    a11y_index = load_a11y_data()
    print(f"Loaded {len(a11y_index)} a11y entries\n")

    has_gt, matched, distinctive = analyze_a11y_coverage(hard_ids, k10_results, a11y_index)
    analyze_element_disambiguation(hard_ids, k10_results, a11y_index)
    analyze_element_id_bypass(hard_ids, k10_results, a11y_index)

    # Save summary
    output_dir = PROJECT_ROOT / "outputs" / "eval_c5"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_hard_cases": len(hard_ids),
        "n_matched_a11y": matched,
        "n_gt_element_found": has_gt,
        "n_distinctive_label": distinctive,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
