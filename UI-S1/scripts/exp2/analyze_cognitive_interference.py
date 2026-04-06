"""Analyze Cognitive Interference experiment results.

Compares 4 conditions:
  A: Isolated UI State (screenshot → describe UI)
  B: Isolated Action Planning (GT UI state + history → predict action, no screenshot)
  C: Standard AR baseline (from existing greedy results)
  D: Chained (model-generated UI state from A → B input)

Key hypotheses:
  - If B ≈ C: planning quality maintained without screenshot → failures from capability limits
  - If B >> C: removing screenshot helps → confirms cognitive interference
  - D vs B gap: measures UI understanding degradation when model generates its own description
  - D vs C: if D ≈ C, chaining preserves quality → interference is real but UI extraction compensates
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load_results(results_path):
    """Load cognitive interference results JSON."""
    with open(results_path) as f:
        return json.load(f)


def compute_step_accuracy(results, condition_key):
    """Compute step accuracy for a condition."""
    successes = 0
    total = 0
    for r in results:
        cond = r.get(condition_key, {})
        if "success" in cond:
            total += 1
            if cond["success"]:
                successes += 1
    return successes / total if total else 0, successes, total


def compute_match_rates(results, condition_key):
    """Compute function/args/status match rates."""
    fn_match = 0
    args_match = 0
    status_match = 0
    total = 0
    for r in results:
        cond = r.get(condition_key, {})
        if "function_match" in cond:
            total += 1
            if cond["function_match"]:
                fn_match += 1
            if cond["args_match"]:
                args_match += 1
            if cond["status_match"]:
                status_match += 1
    if total == 0:
        return 0, 0, 0, 0
    return fn_match / total, args_match / total, status_match / total, total


def analyze_by_domain(results, conditions):
    """Break down accuracy by domain."""
    domains = sorted(set(r["domain"] for r in results))
    table = {}
    for domain in domains:
        domain_results = [r for r in results if r["domain"] == domain]
        row = {"n": len(domain_results)}
        for cond_key in conditions:
            acc, succ, tot = compute_step_accuracy(domain_results, cond_key)
            row[cond_key] = {"accuracy": acc, "correct": succ, "total": tot}
        table[domain] = row
    return table


def analyze_by_position(results, conditions):
    """Break down accuracy by position bucket."""
    buckets = ["early", "mid", "late"]
    table = {}
    for pos in buckets:
        pos_results = [r for r in results if r.get("position_bucket") == pos]
        row = {"n": len(pos_results)}
        for cond_key in conditions:
            acc, succ, tot = compute_step_accuracy(pos_results, cond_key)
            row[cond_key] = {"accuracy": acc, "correct": succ, "total": tot}
        table[pos] = row
    return table


def analyze_by_greedy_correctness(results, conditions):
    """Break down by whether greedy AR got the step right."""
    table = {}
    for label, filter_fn in [("greedy_correct", lambda r: r.get("greedy_correct", False)),
                              ("greedy_wrong", lambda r: not r.get("greedy_correct", False))]:
        subset = [r for r in results if filter_fn(r)]
        row = {"n": len(subset)}
        for cond_key in conditions:
            acc, succ, tot = compute_step_accuracy(subset, cond_key)
            row[cond_key] = {"accuracy": acc, "correct": succ, "total": tot}
        table[label] = row
    return table


def paired_step_comparison(results, cond_a_key, cond_b_key):
    """Paired comparison: on which steps does cond_a beat cond_b and vice versa."""
    both_right = 0
    a_only = 0
    b_only = 0
    both_wrong = 0
    total = 0

    for r in results:
        a_ok = r.get(cond_a_key, {}).get("success", False)
        b_ok = r.get(cond_b_key, {}).get("success", False)
        total += 1
        if a_ok and b_ok:
            both_right += 1
        elif a_ok and not b_ok:
            a_only += 1
        elif not a_ok and b_ok:
            b_only += 1
        else:
            both_wrong += 1

    return {
        "total": total,
        "both_right": both_right,
        "a_only_right": a_only,
        "b_only_right": b_only,
        "both_wrong": both_wrong,
        "agreement": (both_right + both_wrong) / total if total else 0,
    }


def analyze_condition_a_quality(results):
    """Analyze UI state description quality statistics."""
    descs = [r.get("condition_a", {}).get("ui_state_description", "") for r in results]
    descs = [d for d in descs if d]

    if not descs:
        return {"n": 0}

    lengths = [len(d) for d in descs]
    word_counts = [len(d.split()) for d in descs]

    return {
        "n": len(descs),
        "avg_char_length": np.mean(lengths),
        "avg_word_count": np.mean(word_counts),
        "median_word_count": np.median(word_counts),
        "min_word_count": min(word_counts),
        "max_word_count": max(word_counts),
    }


def format_table(header, rows, col_widths=None):
    """Format a simple text table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [header] + rows) + 2
                      for i in range(len(header))]

    sep = "|" + "|".join("-" * w for w in col_widths) + "|"
    fmt_row = lambda row: "|" + "|".join(str(row[i]).center(w) for i, w in enumerate(col_widths)) + "|"

    lines = [fmt_row(header), sep]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def write_report(results, output_dir):
    """Write comprehensive analysis report."""
    conditions = ["condition_b", "condition_c", "condition_d"]
    cond_labels = {
        "condition_b": "B (GT UI→Action)",
        "condition_c": "C (Standard AR)",
        "condition_d": "D (Chained A→B)",
    }

    lines = []
    lines.append("# Exp2: Cognitive Interference Hypothesis — Analysis Report")
    lines.append("")
    lines.append(f"Model: gui360_full_sft_v2 (Qwen2.5-VL-7B) | {len(results)} steps from 202 Pattern B trajectories")
    lines.append("")

    # ---- Table 1: Overall Step Accuracy ----
    lines.append("## 1. Overall Step Accuracy")
    lines.append("")
    lines.append("| Condition | Step Accuracy | Correct | Total | Δ vs C |")
    lines.append("|---|:---:|:---:|:---:|:---:|")

    c_acc, c_succ, c_tot = compute_step_accuracy(results, "condition_c")
    for cond_key in conditions:
        acc, succ, tot = compute_step_accuracy(results, cond_key)
        delta = acc - c_acc
        delta_str = f"{delta:+.4f}" if cond_key != "condition_c" else "—"
        lines.append(f"| {cond_labels[cond_key]} | {acc:.4f} | {succ} | {tot} | {delta_str} |")

    lines.append("")

    # Hypothesis verdict
    b_acc, _, _ = compute_step_accuracy(results, "condition_b")
    d_acc, _, _ = compute_step_accuracy(results, "condition_d")

    lines.append("### Hypothesis Interpretation")
    lines.append("")
    if b_acc > c_acc + 0.02:
        lines.append(f"**B > C by {b_acc - c_acc:.4f}**: Removing the screenshot helps → supports cognitive interference hypothesis.")
    elif abs(b_acc - c_acc) <= 0.02:
        lines.append(f"**B ≈ C (Δ={b_acc - c_acc:.4f})**: Action planning quality is similar with/without screenshot → failures are likely capability limits, not interference.")
    else:
        lines.append(f"**B < C by {c_acc - b_acc:.4f}**: Text-only action planning is worse → the screenshot provides necessary grounding information, not interference.")

    lines.append("")
    if d_acc < b_acc - 0.02:
        lines.append(f"**D < B by {b_acc - d_acc:.4f}**: Chaining model-generated UI state degrades quality → UI state extraction introduces errors.")
    elif abs(d_acc - b_acc) <= 0.02:
        lines.append(f"**D ≈ B (Δ={d_acc - b_acc:.4f})**: Model-generated UI state is as good as GT → reliable scene understanding.")

    lines.append("")

    # ---- Table 2: Match Rate Breakdown ----
    lines.append("## 2. Match Rate Breakdown")
    lines.append("")
    lines.append("| Condition | Function Match | Args Match | Status Match |")
    lines.append("|---|:---:|:---:|:---:|")
    for cond_key in conditions:
        fn, ar, st, tot = compute_match_rates(results, cond_key)
        lines.append(f"| {cond_labels[cond_key]} | {fn:.4f} | {ar:.4f} | {st:.4f} |")

    lines.append("")

    # ---- Table 3: Domain Breakdown ----
    lines.append("## 3. Domain Breakdown")
    lines.append("")
    domain_table = analyze_by_domain(results, conditions)
    lines.append("| Domain | N | B Acc | C Acc | D Acc | B-C Δ | D-C Δ |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for domain in sorted(domain_table.keys()):
        row = domain_table[domain]
        b_a = row.get("condition_b", {}).get("accuracy", 0)
        c_a = row.get("condition_c", {}).get("accuracy", 0)
        d_a = row.get("condition_d", {}).get("accuracy", 0)
        lines.append(f"| {domain} | {row['n']} | {b_a:.4f} | {c_a:.4f} | {d_a:.4f} | {b_a - c_a:+.4f} | {d_a - c_a:+.4f} |")

    lines.append("")

    # ---- Table 4: Position Breakdown ----
    lines.append("## 4. Position Breakdown (Early/Mid/Late)")
    lines.append("")
    pos_table = analyze_by_position(results, conditions)
    lines.append("| Position | N | B Acc | C Acc | D Acc | B-C Δ |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for pos in ["early", "mid", "late"]:
        if pos not in pos_table:
            continue
        row = pos_table[pos]
        b_a = row.get("condition_b", {}).get("accuracy", 0)
        c_a = row.get("condition_c", {}).get("accuracy", 0)
        d_a = row.get("condition_d", {}).get("accuracy", 0)
        lines.append(f"| {pos} | {row['n']} | {b_a:.4f} | {c_a:.4f} | {d_a:.4f} | {b_a - c_a:+.4f} |")

    lines.append("")

    # ---- Table 5: Greedy Correctness Breakdown ----
    lines.append("## 5. Breakdown by Greedy Correctness")
    lines.append("")
    lines.append("> Do conditions B/D help more on steps that greedy AR already got right, or on steps it got wrong?")
    lines.append("")
    gc_table = analyze_by_greedy_correctness(results, conditions)
    lines.append("| Subset | N | B Acc | C Acc | D Acc | B-C Δ |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for label in ["greedy_correct", "greedy_wrong"]:
        if label not in gc_table:
            continue
        row = gc_table[label]
        b_a = row.get("condition_b", {}).get("accuracy", 0)
        c_a = row.get("condition_c", {}).get("accuracy", 0)
        d_a = row.get("condition_d", {}).get("accuracy", 0)
        lines.append(f"| {label} | {row['n']} | {b_a:.4f} | {c_a:.4f} | {d_a:.4f} | {b_a - c_a:+.4f} |")

    lines.append("")

    # ---- Table 6: Paired Step Comparison ----
    lines.append("## 6. Paired Step Comparisons")
    lines.append("")

    for cond_a, cond_b, label in [
        ("condition_b", "condition_c", "B vs C"),
        ("condition_d", "condition_c", "D vs C"),
        ("condition_d", "condition_b", "D vs B"),
    ]:
        comp = paired_step_comparison(results, cond_a, cond_b)
        a_label = cond_labels.get(cond_a, cond_a)
        b_label = cond_labels.get(cond_b, cond_b)
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"| Outcome | Count | % |")
        lines.append(f"|---|:---:|:---:|")
        t = comp["total"]
        lines.append(f"| Both correct | {comp['both_right']} | {comp['both_right']/t*100:.1f}% |")
        lines.append(f"| {a_label.split('(')[0].strip()} only | {comp['a_only_right']} | {comp['a_only_right']/t*100:.1f}% |")
        lines.append(f"| {b_label.split('(')[0].strip()} only | {comp['b_only_right']} | {comp['b_only_right']/t*100:.1f}% |")
        lines.append(f"| Both wrong | {comp['both_wrong']} | {comp['both_wrong']/t*100:.1f}% |")
        lines.append(f"| Agreement | — | {comp['agreement']*100:.1f}% |")
        lines.append("")

    # ---- Table 7: Condition A Quality ----
    lines.append("## 7. Condition A — UI State Description Quality")
    lines.append("")
    a_stats = analyze_condition_a_quality(results)
    if a_stats["n"] > 0:
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|:---:|")
        lines.append(f"| Descriptions generated | {a_stats['n']} |")
        lines.append(f"| Avg character length | {a_stats['avg_char_length']:.0f} |")
        lines.append(f"| Avg word count | {a_stats['avg_word_count']:.1f} |")
        lines.append(f"| Median word count | {a_stats['median_word_count']:.0f} |")
        lines.append(f"| Min/Max word count | {a_stats['min_word_count']} / {a_stats['max_word_count']} |")
    else:
        lines.append("No Condition A descriptions found.")

    lines.append("")

    # ---- Summary ----
    lines.append("## 8. Summary & Key Takeaways")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|:---:|")
    lines.append(f"| Total steps | {len(results)} |")
    lines.append(f"| B (GT UI→Action) accuracy | {b_acc:.4f} |")
    lines.append(f"| C (Standard AR) accuracy | {c_acc:.4f} |")
    lines.append(f"| D (Chained A→B) accuracy | {d_acc:.4f} |")
    lines.append(f"| B - C delta | {b_acc - c_acc:+.4f} |")
    lines.append(f"| D - C delta | {d_acc - c_acc:+.4f} |")
    lines.append(f"| D - B delta (UI extraction gap) | {d_acc - b_acc:+.4f} |")
    lines.append("")

    report = "\n".join(lines)

    # Save
    report_path = os.path.join(output_dir, "COGNITIVE_INTERFERENCE_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Also save raw stats as JSON
    stats = {
        "overall": {
            cond_key: {
                "accuracy": compute_step_accuracy(results, cond_key)[0],
                "correct": compute_step_accuracy(results, cond_key)[1],
                "total": compute_step_accuracy(results, cond_key)[2],
            }
            for cond_key in conditions
        },
        "by_domain": {
            domain: {
                cond_key: row[cond_key]
                for cond_key in conditions if cond_key in row
            }
            for domain, row in domain_table.items()
        },
        "by_position": {
            pos: {
                cond_key: row[cond_key]
                for cond_key in conditions if cond_key in row
            }
            for pos, row in pos_table.items()
        },
        "condition_a_quality": a_stats,
    }
    stats_path = os.path.join(output_dir, "cognitive_interference_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats saved to {stats_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze Cognitive Interference experiment")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing cognitive_interference_results.json")
    parser.add_argument("--condition_ab_results", type=str, default=None,
                        help="Path to condition A+B results (if run separately from D)")
    parser.add_argument("--condition_d_results", type=str, default=None,
                        help="Path to condition D results (if run separately)")
    args = parser.parse_args()

    # Load results — support both single file and split files
    results_path = os.path.join(args.results_dir, "cognitive_interference_results.json")

    if os.path.exists(results_path):
        data = load_results(results_path)
        results = data["results"]
        print(f"Loaded {len(results)} step results from {results_path}")
    else:
        # Try merging separate condition files
        if args.condition_ab_results and args.condition_d_results:
            ab_data = load_results(args.condition_ab_results)
            d_data = load_results(args.condition_d_results)

            # Merge: ab has condition_a, condition_b, condition_c; d has condition_d
            ab_results = {r["sample_id"]: r for r in ab_data["results"]}
            for r in d_data["results"]:
                if r["sample_id"] in ab_results:
                    ab_results[r["sample_id"]]["condition_d"] = r.get("condition_d", {})

            results = list(ab_results.values())
            print(f"Merged {len(results)} results from separate condition files")
        else:
            # Try to find individual condition result files
            merged = {}
            for fname in sorted(os.listdir(args.results_dir)):
                if fname.endswith("_results.json") and "cognitive" in fname:
                    fpath = os.path.join(args.results_dir, fname)
                    data = load_results(fpath)
                    for r in data["results"]:
                        sid = r["sample_id"]
                        if sid not in merged:
                            merged[sid] = r
                        else:
                            # Merge condition keys
                            for k, v in r.items():
                                if k.startswith("condition_"):
                                    merged[sid][k] = v

            results = list(merged.values())
            if not results:
                print(f"No results found in {args.results_dir}")
                return
            print(f"Merged {len(results)} results from directory")

    # Write report
    write_report(results, args.results_dir)


if __name__ == "__main__":
    main()
