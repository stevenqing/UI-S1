#!/usr/bin/env python3
"""Generate Base vs SFT v3 comparison report from exp2 results."""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR / "results"
SFT_DIR = SCRIPT_DIR / "results" / "sft_v3"
OUTPUT_FILE = SCRIPT_DIR / "results" / "BASE_VS_SFT_COMPARISON.md"


def load_json(path):
    """Load JSON file, return None if not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARNING: Could not load {path}: {e}")
        return None


def load_gui360_summary(results_dir, link_name):
    """Load GUI-360 evaluation summary from latest symlink."""
    latest = Path(results_dir) / link_name
    if not latest.exists():
        return None
    # Find summary JSON in the directory
    for f in sorted(latest.iterdir()):
        if "summary" in f.name and f.suffix == ".json":
            return load_json(f)
    return None


def fmt_pct(val, decimals=2):
    """Format as percentage string."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}%"


def fmt_delta(base_val, sft_val, decimals=2):
    """Format delta with sign."""
    if base_val is None or sft_val is None:
        return "N/A"
    delta = sft_val - base_val
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{decimals}f}pp"


def main():
    report_lines = []

    def add(line=""):
        report_lines.append(line)

    add("# Exp2: Base Model vs SFT v3 Comparison Report")
    add()
    add("Model comparison across all 4 experimental directions.")
    add()
    add("- **Base**: Qwen2.5-VL-7B-Instruct")
    add("- **SFT v3**: gui360_lora_sft_v3_merged (fine-tuned on GUI-360)")
    add()
    add("---")
    add()

    # ============================================================
    # Section 1: Error Cascade (Step 1)
    # ============================================================
    add("## 1. Error Cascade Analysis (Step 1)")
    add()

    base_ac_cascade = load_json(BASE_DIR / "analysis" / "ac_cascade_metrics.json")
    sft_ac_cascade = load_json(SFT_DIR / "analysis" / "ac_cascade_metrics.json")
    base_g360_cascade = load_json(BASE_DIR / "analysis" / "gui360_cascade_metrics.json")
    sft_g360_cascade = load_json(SFT_DIR / "analysis" / "gui360_cascade_metrics.json")

    add("### AndroidControl (Cross-Domain Transfer)")
    add()
    if base_ac_cascade and sft_ac_cascade:
        add("| Metric | Base | SFT v3 | Delta |")
        add("|--------|:---:|:---:|:---:|")
        for key, label in [
            ("trajectory_success_rate", "TSR"),
            ("step0_failure_rate", "Step-0 Failure Rate"),
            ("mean_first_error_position", "Mean First Error Pos"),
            ("mean_cascade_depth", "Mean Cascade Depth"),
            ("post_error_accuracy", "Post-Error Accuracy"),
        ]:
            bv = base_ac_cascade.get(key)
            sv = sft_ac_cascade.get(key)
            if key == "mean_first_error_position" or key == "mean_cascade_depth":
                add(f"| {label} | {bv:.4f} | {sv:.4f} | {sv-bv:+.4f} |")
            else:
                add(f"| {label} | {fmt_pct(bv)} | {fmt_pct(sv)} | {fmt_delta(bv, sv)} |")
    else:
        add("*Results not yet available.*")
    add()

    add("### GUI-360 (In-Domain)")
    add()
    if base_g360_cascade and sft_g360_cascade:
        add("| Metric | Base | SFT v3 | Delta |")
        add("|--------|:---:|:---:|:---:|")
        for key, label in [
            ("trajectory_success_rate", "TSR"),
            ("step0_failure_rate", "Step-0 Failure Rate"),
            ("mean_first_error_position", "Mean First Error Pos"),
            ("mean_cascade_depth", "Mean Cascade Depth"),
            ("post_error_accuracy", "Post-Error Accuracy"),
        ]:
            bv = base_g360_cascade.get(key)
            sv = sft_g360_cascade.get(key)
            if key == "mean_first_error_position" or key == "mean_cascade_depth":
                add(f"| {label} | {bv:.4f} | {sv:.4f} | {sv-bv:+.4f} |")
            else:
                add(f"| {label} | {fmt_pct(bv)} | {fmt_pct(sv)} | {fmt_delta(bv, sv)} |")
    else:
        add("*Results not yet available.*")
    add()

    # Survival probability comparison
    if base_g360_cascade and sft_g360_cascade:
        add("### Survival Probability P(step k+1 correct | step k correct)")
        add()
        add("| Step k | Base AC | SFT AC | Base G360 | SFT G360 |")
        add("|:---:|:---:|:---:|:---:|:---:|")
        for k in range(6):
            sk = str(k)
            b_ac = base_ac_cascade["survival_probability"].get(sk, None) if base_ac_cascade else None
            s_ac = sft_ac_cascade["survival_probability"].get(sk, None) if sft_ac_cascade else None
            b_g = base_g360_cascade["survival_probability"].get(sk, None) if base_g360_cascade else None
            s_g = sft_g360_cascade["survival_probability"].get(sk, None) if sft_g360_cascade else None
            row = f"| {k} |"
            for v in [b_ac, s_ac, b_g, s_g]:
                row += f" {v:.3f} |" if v is not None else " N/A |"
            add(row)
        add()

    # Oracle rescue comparison
    add("### Oracle Rescue (AC)")
    add()
    base_ac_natural = load_json(BASE_DIR / "ac" / "ac_nostop_natural_cascade_summary.json")
    base_ac_oracle = load_json(BASE_DIR / "ac" / "ac_nostop_oracle_rescue_summary.json")
    sft_ac_natural = load_json(SFT_DIR / "ac" / "ac_nostop_natural_cascade_summary.json")
    sft_ac_oracle = load_json(SFT_DIR / "ac" / "ac_nostop_oracle_rescue_summary.json")

    if sft_ac_natural and sft_ac_oracle:
        add("| | Base Natural | Base Oracle | SFT Natural | SFT Oracle |")
        add("|---|:---:|:---:|:---:|:---:|")
        # TSR
        bn_tsr = base_ac_natural.get("trajectory_success_rate", None) if base_ac_natural else None
        bo_tsr = base_ac_oracle.get("trajectory_success_rate", None) if base_ac_oracle else None
        sn_tsr = sft_ac_natural.get("trajectory_success_rate", None)
        so_tsr = sft_ac_oracle.get("trajectory_success_rate", None)
        add(f"| TSR | {fmt_pct(bn_tsr)} | {fmt_pct(bo_tsr)} | {fmt_pct(sn_tsr)} | {fmt_pct(so_tsr)} |")
        # Scattered progress
        bn_sp = base_ac_natural.get("scattered_progress_rate", None) if base_ac_natural else None
        bo_sp = base_ac_oracle.get("scattered_progress_rate", None) if base_ac_oracle else None
        sn_sp = sft_ac_natural.get("scattered_progress_rate", None)
        so_sp = sft_ac_oracle.get("scattered_progress_rate", None)
        add(f"| Scattered Progress | {bn_sp:.4f if bn_sp else 'N/A'} | {bo_sp:.4f if bo_sp else 'N/A'} | {sn_sp:.4f if sn_sp else 'N/A'} | {so_sp:.4f if so_sp else 'N/A'} |")
    else:
        add("*SFT v3 oracle rescue results not yet available.*")
    add()

    add("---")
    add()

    # ============================================================
    # Section 2: Error Classification (Step 2)
    # ============================================================
    add("## 2. LLM Error Classification (Step 2)")
    add()

    base_cls = load_json(BASE_DIR / "classification" / "classification_summary.json")
    sft_cls = load_json(SFT_DIR / "classification" / "classification_summary.json")

    if base_cls and sft_cls:
        add("### Cross-Dataset Comparison")
        add()
        add("| Category | Base AC | SFT AC | Base G360 | SFT G360 |")
        add("|----------|:---:|:---:|:---:|:---:|")
        for cat in ["GROUNDING", "VOCABULARY", "PLANNING", "CONTEXT"]:
            b_ac = base_cls.get("ac", {}).get("category_percentages", {}).get(cat, None)
            s_ac = sft_cls.get("ac", {}).get("category_percentages", {}).get(cat, None)
            b_g = base_cls.get("gui360", {}).get("category_percentages", {}).get(cat, None)
            s_g = sft_cls.get("gui360", {}).get("category_percentages", {}).get(cat, None)
            add(f"| **{cat.capitalize()}** | {fmt_pct(b_ac, 1)} | {fmt_pct(s_ac, 1)} | {fmt_pct(b_g, 1)} | {fmt_pct(s_g, 1)} |")

        # Total errors
        b_ac_n = base_cls.get("ac", {}).get("total_errors", "?")
        s_ac_n = sft_cls.get("ac", {}).get("total_errors", "?")
        b_g_n = base_cls.get("gui360", {}).get("total_errors", "?")
        s_g_n = sft_cls.get("gui360", {}).get("total_errors", "?")
        add(f"| *N (sampled)* | *{b_ac_n}* | *{s_ac_n}* | *{b_g_n}* | *{s_g_n}* |")
    else:
        add("*Classification results not yet available.*")
    add()

    add("---")
    add()

    # ============================================================
    # Section 3: Summary Context (Step 3)
    # ============================================================
    add("## 3. Summary Context Evaluation (Step 3)")
    add()

    add("### AndroidControl")
    add()
    add("| Format | Base TSR | SFT TSR | Base Progress | SFT Progress |")
    add("|--------|:---:|:---:|:---:|:---:|")

    for fmt in ["action_level", "semantic_level", "progress_level"]:
        base_f = load_json(BASE_DIR / "ac" / f"ac_summary_{fmt}_summary.json")
        sft_f = load_json(SFT_DIR / "ac" / f"ac_summary_{fmt}_summary.json")
        b_tsr = base_f.get("trajectory_success_rate", None) if base_f else None
        s_tsr = sft_f.get("trajectory_success_rate", None) if sft_f else None
        b_prog = base_f.get("avg_progress", None) if base_f else None
        s_prog = sft_f.get("avg_progress", None) if sft_f else None
        add(f"| {fmt} | {fmt_pct(b_tsr)} | {fmt_pct(s_tsr)} | {b_prog:.4f if b_prog else 'N/A'} | {s_prog:.4f if s_prog else 'N/A'} |")

    # AR baseline reference
    if base_ac_cascade and sft_ac_cascade:
        b_tsr = base_ac_cascade.get("trajectory_success_rate")
        s_tsr = sft_ac_cascade.get("trajectory_success_rate")
        add(f"| *AR baseline (ref)* | *{fmt_pct(b_tsr)}* | *{fmt_pct(s_tsr)}* | — | — |")
    add()

    add("### GUI-360")
    add()

    base_g_nostop_sum = load_gui360_summary(BASE_DIR / "gui360", "latest_nostop")
    sft_g_nostop_sum = load_gui360_summary(SFT_DIR / "gui360", "latest_nostop")
    base_g_summary_sum = load_gui360_summary(BASE_DIR / "gui360", "latest_summary")
    sft_g_summary_sum = load_gui360_summary(SFT_DIR / "gui360", "latest_summary")
    base_g_subtask_sum = load_gui360_summary(BASE_DIR / "gui360", "latest_subtask")
    sft_g_subtask_sum = load_gui360_summary(SFT_DIR / "gui360", "latest_subtask")

    add("| Mode | Base TSR | SFT TSR | Base Step Acc | SFT Step Acc |")
    add("|------|:---:|:---:|:---:|:---:|")

    for label, base_sum, sft_sum in [
        ("No-stop AR", base_g_nostop_sum, sft_g_nostop_sum),
        ("Summary (stop)", base_g_summary_sum, sft_g_summary_sum),
        ("Subtask oracle (stop)", base_g_subtask_sum, sft_g_subtask_sum),
    ]:
        b_tsr = base_sum.get("trajectory_success_rate", None) if base_sum else None
        s_tsr = sft_sum.get("trajectory_success_rate", None) if sft_sum else None
        b_sa = base_sum.get("step_accuracy", base_sum.get("overall_accuracy", None)) if base_sum else None
        s_sa = sft_sum.get("step_accuracy", sft_sum.get("overall_accuracy", None)) if sft_sum else None
        # Convert to percentage if needed
        if b_sa is not None and b_sa <= 1.0:
            b_sa *= 100
        if s_sa is not None and s_sa <= 1.0:
            s_sa *= 100
        add(f"| {label} | {fmt_pct(b_tsr)} | {fmt_pct(s_tsr)} | {fmt_pct(b_sa)} | {fmt_pct(s_sa)} |")
    add()

    add("---")
    add()

    # ============================================================
    # Section 4: Subtask Decomposition (Step 4)
    # ============================================================
    add("## 4. Subtask Decomposition (Step 4 - AC)")
    add()

    base_ac_sub = load_json(BASE_DIR / "ac" / "ac_subtask_eval_summary.json")
    sft_ac_sub = load_json(SFT_DIR / "ac" / "ac_subtask_eval_summary.json")

    if base_ac_sub and sft_ac_sub:
        add("| Metric | Base | SFT v3 | Delta |")
        add("|--------|:---:|:---:|:---:|")
        for key, label in [
            ("trajectory_success_rate", "TSR"),
            ("step_accuracy", "Step Accuracy"),
            ("subtask_completion_rate", "Subtask Completion Rate"),
        ]:
            bv = base_ac_sub.get(key)
            sv = sft_ac_sub.get(key)
            if bv is not None and sv is not None:
                if bv <= 1.0:
                    bv *= 100
                if sv <= 1.0:
                    sv *= 100
                add(f"| {label} | {fmt_pct(bv)} | {fmt_pct(sv)} | {fmt_delta(bv, sv)} |")
            else:
                add(f"| {label} | {bv} | {sv} | — |")
    else:
        add("*Subtask decomposition results not yet available.*")
    add()

    add("---")
    add()

    # ============================================================
    # Synthesis
    # ============================================================
    add("## Synthesis: Base vs SFT v3")
    add()
    add("| Finding | Base | SFT v3 | Improvement? |")
    add("|---------|:---:|:---:|:---:|")

    # TSR comparison
    if base_g360_cascade and sft_g360_cascade:
        b = base_g360_cascade["trajectory_success_rate"]
        s = sft_g360_cascade["trajectory_success_rate"]
        add(f"| GUI-360 TSR (no-stop) | {fmt_pct(b)} | {fmt_pct(s)} | {fmt_delta(b, s)} |")

    if base_ac_cascade and sft_ac_cascade:
        b = base_ac_cascade["trajectory_success_rate"]
        s = sft_ac_cascade["trajectory_success_rate"]
        add(f"| AC TSR (no-stop) | {fmt_pct(b)} | {fmt_pct(s)} | {fmt_delta(b, s)} |")

    # Step-0 failure
    if base_g360_cascade and sft_g360_cascade:
        b = base_g360_cascade["step0_failure_rate"]
        s = sft_g360_cascade["step0_failure_rate"]
        add(f"| GUI-360 Step-0 Fail | {fmt_pct(b)} | {fmt_pct(s)} | {fmt_delta(b, s)} |")

    if base_ac_cascade and sft_ac_cascade:
        b = base_ac_cascade["step0_failure_rate"]
        s = sft_ac_cascade["step0_failure_rate"]
        add(f"| AC Step-0 Fail | {fmt_pct(b)} | {fmt_pct(s)} | {fmt_delta(b, s)} |")

    # Post-error accuracy
    if base_g360_cascade and sft_g360_cascade:
        b = base_g360_cascade["post_error_accuracy"]
        s = sft_g360_cascade["post_error_accuracy"]
        add(f"| GUI-360 Post-Error Acc | {fmt_pct(b)} | {fmt_pct(s)} | {fmt_delta(b, s)} |")

    add()
    add("### Key Takeaways")
    add()
    add("*(To be filled after results are available)*")
    add()

    # Write report
    report = "\n".join(report_lines)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(report)

    print(f"Report written to: {OUTPUT_FILE}")
    print()
    print(report)


if __name__ == "__main__":
    main()
