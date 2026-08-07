import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} pp"


def interval(values):
    return f"[{100 * values[0]:+.2f}, {100 * values[1]:+.2f}] pp"


def build_report(xf, mde):
    evaluations = xf["evaluations"]
    comparisons = xf["comparisons"]
    rank = xf["proposer_rank_containment"]
    budget = xf["budget_matching"]
    lines = [
        "# Cross-Benchmark Q1 Transfer 总结",
        "",
        "日期：2026-08-07",
        "",
        f"状态：`{'MIND2WEB_PASS' if xf['XF1'] else 'MIND2WEB_XF1_NOT_MET'}`",
        "",
        "## Mind2Web 主结果",
        "",
        "| Arm | Micro Step SR | Episode macro Step SR | Mean forwards |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("C_uni", "C_cond", "C_rand", "C_self"):
        value = evaluations[arm]
        lines.append(
            f"| {arm} | {percent(value['micro_step_sr'])} | "
            f"{percent(value['episode_macro_step_sr'])} | {value['mean_forwards']:.3f} |"
        )
    lines.extend(["", "## 配对比较", ""])
    for reference in ("C_uni", "C_rand", "C_self"):
        value = comparisons[reference]
        lines.append(
            f"- C-cond − {reference}: {pp(value['point_delta'])}, "
            f"99% CI {interval(value['ci_99'])}."
        )
    trigger_rate = evaluations["C_cond"]["triggered_rows"] / xf["rows"]
    lines.extend([
        "",
        "## 预注册裁决",
        "",
        f"- XF1: **{xf['XF1']}**；Mind2Web MDE = {pp(mde['micro_mde'])}.",
        f"- XF2: **{xf['XF2']}**.",
        f"- XF4: **{xf['XF4']}**.",
        f"- XF-K1: **{xf['XF_K1']}**.",
        f"- XF-K2: **{xf['XF_K2']}**.",
        f"- XF-K3: **{xf['XF_K3']}**；stage-2 trigger rate = {percent(trigger_rate)}.",
        f"- AndroidControl decision: **{'CANCEL_XF_K1' if xf['XF_K1'] else 'PROCEED'}**.",
        "",
        "## 必报诊断",
        "",
        f"- Triggered C-cond Step SR: {percent(evaluations['C_cond']['triggered_micro_step_sr']) if evaluations['C_cond']['triggered_micro_step_sr'] is not None else 'N/A'}.",
        f"- Non-triggered C-cond Step SR: {percent(evaluations['C_cond']['nontriggered_micro_step_sr']) if evaluations['C_cond']['nontriggered_micro_step_sr'] is not None else 'N/A'}.",
        f"- Rank-0 full-bbox containment: {percent(rank['rank0_full_bbox_containment'])}.",
        f"- Mean rank0–11 full-bbox containment: {percent(rank['mean_rank0_to_rank11_full_bbox_containment'])}.",
        f"- Single-cluster geometry fallback rows: {xf['cluster_fallback_rows']}.",
        f"- Max arm mean-forward difference: {budget['max_arm_mean_difference']:.3f}; budget-matched control required: {budget['required']}.",
        "",
        "## 产物",
        "",
        "- `mde_mind2web.json`",
        "- `xf_mind2web.json`",
        "- `STATUS.json`",
        "- `raw/mind2web-consensus-roi.jsonl`",
        "- `/scratch/workspaceblobstore/xfer-traces/2026-08-07/BACKUP_MANIFEST.json`",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xf", type=Path, default=RUN_DIR / "xf_mind2web.json")
    parser.add_argument("--mde", type=Path, default=RUN_DIR / "mde_mind2web.json")
    args = parser.parse_args()
    xf = json.loads(args.xf.read_text())
    mde = json.loads(args.mde.read_text())
    if xf["status"] != "PASS" or mde["status"] != "PASS":
        raise ValueError("finalization requires PASS XF and MDE artifacts")
    report_path = RUN_DIR / "CONSOLIDATED_SUMMARY_ZH.md"
    report_path.write_text(build_report(xf, mde) + "\n")
    sources = {
        "xf_mind2web.json": sha256_file(args.xf),
        "mde_mind2web.json": sha256_file(args.mde),
        "CONSOLIDATED_SUMMARY_ZH.md": sha256_file(report_path),
    }
    status = {
        "schema_version": 1,
        "status": "MIND2WEB_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "XF1": xf["XF1"],
        "XF2": xf["XF2"],
        "XF4": xf["XF4"],
        "XF_K1": xf["XF_K1"],
        "XF_K2": xf["XF_K2"],
        "XF_K3": xf["XF_K3"],
        "androidcontrol_decision": "CANCEL_XF_K1" if xf["XF_K1"] else "PROCEED",
        "sources": sources,
    }
    (RUN_DIR / "STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, sort_keys=True))


def test_contracts():
    assert percent(0.5) == "50.00%"
    assert pp(0.0123) == "+1.23 pp"
    assert interval([-0.01, 0.02]) == "[-1.00, +2.00] pp"


if __name__ == "__main__":
    test_contracts()
    if len(__import__("sys").argv) > 1:
        main()
    else:
        print(json.dumps({"status": "PASS"}))