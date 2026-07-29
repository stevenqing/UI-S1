import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
LANES = [
    ("UI-AGILE-3B", "ui-agile-3b", "low"),
    ("UI-AGILE-3B", "ui-agile-3b", "high"),
    ("UI-AGILE-7B", "ui-agile-7b", "low"),
    ("UI-AGILE-7B", "ui-agile-7b", "high"),
    ("UI-R1-E-3B", "ui-r1-e-3b", "low"),
    ("UI-R1-E-3B", "ui-r1-e-3b", "high"),
    ("GUI-R1-3B", "gui-r1-3b", "low"),
    ("GUI-R1-3B", "gui-r1-3b", "high"),
    ("GUI-R1-7B", "gui-r1-7b", "low"),
    ("GUI-R1-7B", "gui-r1-7b", "high"),
]
PAPER_ANCHORS = {
    ("UI-AGILE-3B", "low"): (98.1, 91.8, 90.8),
    ("UI-AGILE-3B", "high"): (88.8, 85.8, 78.7),
    ("UI-AGILE-7B", "low"): (98.0, 92.2, 91.0),
    ("UI-AGILE-7B", "high"): (91.0, 87.2, 81.4),
    ("UI-R1-E-3B", "low"): (97.7, 91.2, 89.4),
    ("UI-R1-E-3B", "high"): (83.5, 78.9, 69.8),
    ("GUI-R1-3B", "low"): (96.9, 89.9, 87.3),
    ("GUI-R1-3B", "high"): (58.0, 56.2, 46.6),
    ("GUI-R1-7B", "low"): (97.5, 91.7, 89.7),
    ("GUI-R1-7B", "high"): (71.6, 65.6, 51.7),
}


def percentage(value: float) -> str:
    return f"{value * 100:.4f}"


def main() -> None:
    manifest = json.loads((RUN_DIR / "artifact_manifest.json").read_text())
    if manifest["status"] != "DOWNLOADED_HASH_INDEX_VERIFIED":
        raise ValueError("checkpoint/data/source manifest is not verified")

    rows = []
    for display_name, artifact_name, setting in LANES:
        root = RUN_DIR / "artifacts" / artifact_name / setting
        score = json.loads((root / "score.json").read_text())
        audit = json.loads((root / "audit.json").read_text())
        if score["coverage"] != "COMPLETE" or score["rows"] != 7708:
            raise ValueError(f"incomplete score: {artifact_name}/{setting}")
        if audit["status"] != "PASS" or audit["rows"] != 7708:
            raise ValueError(f"failed audit: {artifact_name}/{setting}")
        metrics = score["metrics"]
        rows.append({
            "model": display_name,
            "setting": setting.capitalize(),
            "type": metrics["action"]["accuracy"],
            "grounding": metrics["grounding"]["accuracy"],
            "step": metrics["step_success"]["accuracy"],
            "anchor": PAPER_ANCHORS[(display_name, setting)],
            "model_name": audit["model_name"],
            "model_revision": audit["model_revision"],
            "predictions_sha256": audit["predictions_sha256"],
        })

    original = json.loads(
        (RUN_DIR / "original-ui-r1/artifacts/full/score_audit.json").read_text()
    )
    if original["status"] != "PASS" or original["rows"] != 7868:
        raise ValueError("original UI-R1 selected-Low audit is incomplete")

    lines = [
        "# AndroidControl RFT Baseline Results",
        "",
        "更新时间：2026-07-29",
        "",
        "## 1. 统一 7,708-step Low/High lane",
        "",
        "指标顺序为 Type Accuracy / Grounding Accuracy / Step Success Rate。所有结果使用固定官方 parquet、官方 prompt/parser/evaluator、temperature 0、max tokens 256，四卡独立分片、有序合并和独立全量审计；不做 GT output repair。",
        "",
        "| Model | Setting | Type | Grounding | Step SR | Paper anchor | Audit |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        anchor = " / ".join(f"{value:.1f}" for value in row["anchor"])
        lines.append(
            f"| {row['model']} | {row['setting']} | **{percentage(row['type'])}** | "
            f"**{percentage(row['grounding'])}** | **{percentage(row['step'])}** | "
            f"{anchor} | PASS |"
        )
    lines.extend([
        "",
        "## 2. 原始 UI-R1-3B v1 selected-Low lane",
        "",
        "该 lane 使用发布的 7,868-step `ac_test.json`，与上面的 7,708-step lane 不是同一 split。指标为 Type、click Grounding，以及两者算术平均，不含 Step SR。",
        "",
        "| Rows | Episodes | Type | Click Grounding | Reported Average | Paper anchor | Audit |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
        f"| {original['rows']} | {original['episodes']} | **{percentage(original['type_accuracy'])}** | "
        f"**{percentage(original['grounding_accuracy'])}** | **{percentage(original['reported_average'])}** | "
        "94.3 / 82.6 / 88.5 | PASS |",
        "",
        "发布代码存在可复核的坐标矛盾：模型实际 slow-processor grid 为 672x1484，但 `eval_ac.py` 硬编码按 644x1484 缩放。这里按发布 evaluator 复测并在逐行 provenance 中同时记录实际 grid，因此该行标为 released-code controlled reproduction，而不是无保留的 strict paper reproduction。",
        "",
        "## 3. Provenance",
        "",
        f"- Unified artifact manifest: `{(RUN_DIR / 'artifact_manifest.json').relative_to(RUN_DIR.parent.parent.parent)}`",
        f"- Official GCS source manifest: `{(RUN_DIR / 'data/official-gcs/official_gcs_manifest.json').relative_to(RUN_DIR.parent.parent.parent)}`",
        f"- Original UI-R1 image manifest: `{(RUN_DIR / 'original-ui-r1/image_manifest.json').relative_to(RUN_DIR.parent.parent.parent)}`",
        "- Unified checkpoint revisions:",
    ])
    seen = set()
    for row in rows:
        identity = (row["model_name"], row["model_revision"])
        if identity in seen:
            continue
        seen.add(identity)
        lines.append(f"  - `{identity[0]}@{identity[1]}`")
    lines.append(f"- Original UI-R1: `{original['model_name']}@{original['model_revision']}`")
    lines.append("")

    output = RUN_DIR / "ANDROIDCONTROL_RFT_FINAL_REPORT.md"
    output.write_text("\n".join(lines))
    print(json.dumps({
        "status": "PASS",
        "unified_lanes": len(rows),
        "original_lane_rows": original["rows"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()