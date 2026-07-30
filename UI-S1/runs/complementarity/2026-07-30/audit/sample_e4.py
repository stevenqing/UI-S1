import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from common import load_rows, pivot_rows


LABELS = (
    "solvable",
    "annotation_error",
    "partially_observable",
    "ambiguous_instruction",
    "evaluator_artifact",
)


def stable_rank(value: str) -> str:
    return hashlib.sha256(("20260730:" + value).encode()).hexdigest()


def dominant_error(model_rows):
    counts = Counter(row["err_label"] for row in model_rows.values())
    highest = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == highest)
    return winners[0] if len(winners) == 1 else "tie"


def largest_remainder(counts: Counter, total: int) -> dict:
    available = sum(counts.values())
    raw = {key: total * count / available for key, count in counts.items()}
    allocation = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(allocation.values())
    for key in sorted(counts, key=lambda item: (-(raw[item] - allocation[item]), item))[:remaining]:
        allocation[key] += 1
    return allocation


def hard_core_sample(bench, setting, total):
    rows = load_rows(bench, setting)
    identities, models, pivot = pivot_rows(rows)
    hard = []
    for row_id in identities:
        model_rows = pivot[row_id]
        if any(row["success"] for row in model_rows.values()):
            continue
        reference = next(iter(model_rows.values()))
        hard.append((row_id, dominant_error(model_rows), reference, model_rows))
    strata = Counter(item[1] for item in hard)
    allocation = largest_remainder(strata, total)
    selected = []
    for label, count in allocation.items():
        candidates = sorted((item for item in hard if item[1] == label), key=lambda item: stable_rank(item[0]))
        selected.extend(candidates[:count])
    return selected, {"clean_hard_core": len(hard), "strata": dict(strata), "allocation": allocation, "models": models}


def audit_record(stream, item, sample_index):
    row_id, dominant, reference, model_rows = item
    return {
        "sample_id": f"{stream.replace('/', '_')}_{sample_index:03d}",
        "audit_stream": stream,
        "bench": reference["bench"],
        "setting": reference["setting"],
        "row_id": row_id,
        "episode_id": reference["episode_id"],
        "group_key": reference["group_key"],
        "instruction": reference["instruction"],
        "history": reference["history"],
        "image_ref": reference["image_ref"],
        "gt_action": reference["gt_action"],
        "gt_x": reference["gt_x"],
        "gt_y": reference["gt_y"],
        "gt_bbox": reference["gt_bbox"],
        "gt_param": reference["gt_param"],
        "dominant_error": dominant,
        "model_outputs": {
            model: {
                "pred_action": row["pred_action"], "pred_x": row["pred_x"], "pred_y": row["pred_y"],
                "pred_param": row["pred_param"], "pred_raw": row["pred_raw"],
                "err_label": row["err_label"], "success": row["success"],
            }
            for model, row in sorted(model_rows.items())
        },
    }


def select_tasks():
    tasks = []
    manifest = {}
    for bench, setting, total in (
        ("androidcontrol", "low", 100),
        ("androidcontrol", "high", 50),
        ("mind2web", "visual", 100),
    ):
        selected, details = hard_core_sample(bench, setting, total)
        stream = f"e4/{bench}/{setting}"
        tasks.extend(audit_record(stream, item, index) for index, item in enumerate(selected))
        manifest[stream] = details

    visual_rows = load_rows("mind2web", "visual")
    visual_ids, _, visual_pivot = pivot_rows(visual_rows)
    select_ids = [row_id for row_id in visual_ids if next(iter(visual_pivot[row_id].values()))["gt_action"] == "SELECT"]
    for index, row_id in enumerate(sorted(select_ids, key=stable_rank)):
        reference = next(iter(visual_pivot[row_id].values()))
        tasks.append({
            "sample_id": f"d2_select_{index:03d}", "audit_stream": "d2/select_visibility",
            "bench": "mind2web", "setting": "visual", "row_id": row_id,
            "episode_id": reference["episode_id"], "group_key": reference["group_key"],
            "instruction": reference["instruction"], "history": reference["history"],
            "image_ref": reference["image_ref"], "gt_action": reference["gt_action"],
            "gt_bbox": reference["gt_bbox"], "gt_param": reference["gt_param"],
        })
    manifest["d2/select_visibility"] = {"rows": len(select_ids), "sampling": "complete census"}

    html_rows = load_rows("mind2web", "html")
    _, _, html_pivot = pivot_rows(html_rows)
    mindact_only = []
    for row_id in visual_ids:
        visual_success = any(row["success"] for row in visual_pivot[row_id].values())
        html_success = html_pivot[row_id]["mindact-flan-t5-xl"]["success"]
        if html_success and not visual_success:
            reference = next(iter(visual_pivot[row_id].values()))
            mindact_only.append({
                "sample_id": f"cross_modal_mindact_only_{len(mindact_only):03d}",
                "audit_stream": "e4/cross_modal_mindact_only", "bench": "mind2web", "setting": "visual_html",
                "row_id": row_id, "episode_id": reference["episode_id"], "group_key": reference["group_key"],
                "instruction": reference["instruction"], "history": reference["history"],
                "image_ref": reference["image_ref"], "gt_action": reference["gt_action"],
                "gt_bbox": reference["gt_bbox"], "gt_param": reference["gt_param"],
                "visual_outputs": {model: {"pred_raw": row["pred_raw"], "err_label": row["err_label"]}
                                   for model, row in sorted(visual_pivot[row_id].items())},
                "mindact_output": html_pivot[row_id]["mindact-flan-t5-xl"]["pred_raw"],
            })
    tasks.extend(mindact_only)
    manifest["e4/cross_modal_mindact_only"] = {"rows": len(mindact_only), "sampling": "complete census"}
    if len(select_ids) != 79 or len(mindact_only) != 109:
        raise ValueError(f"expected 79 SELECT and 109 MindAct-only rows, found {len(select_ids)} and {len(mindact_only)}")
    return tasks, manifest


def label_assignments(tasks):
    main_tasks = [task for task in tasks if task["audit_stream"].startswith("e4/") and "cross_modal" not in task["audit_stream"]]
    overlap = {task["sample_id"] for task in sorted(main_tasks, key=lambda task: stable_rank(task["sample_id"]))[:30]}
    assignments = []
    for index, task in enumerate(main_tasks):
        annotators = ["annotator_a", "annotator_b"] if task["sample_id"] in overlap else ["annotator_a" if index % 2 == 0 else "annotator_b"]
        for annotator in annotators:
            assignments.append({
                "sample_id": task["sample_id"], "annotator": annotator, "label": None,
                "select_option_visible": None if task["gt_action"] != "SELECT" else None,
                "notes": "",
            })
    for task in tasks:
        if task["audit_stream"] == "d2/select_visibility":
            assignments.append({
                "sample_id": task["sample_id"], "annotator": "annotator_a",
                "label": None, "select_option_visible": None, "notes": "",
            })
    return assignments, sorted(overlap)


def write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    tasks, streams = select_tasks()
    assignments, overlap = label_assignments(tasks)
    write_jsonl(args.sample, tasks)
    write_jsonl(args.labels, assignments)
    manifest = {
        "status": "READY_FOR_HUMAN_ANNOTATION",
        "allowed_labels": LABELS,
        "tasks": len(tasks), "label_assignments": len(assignments),
        "cohen_kappa_overlap_rows": len(overlap), "overlap_sample_ids": overlap,
        "streams": streams,
        "instruction": "Two real annotators must fill assigned rows independently; null labels are intentional.",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "tasks", "label_assignments", "cohen_kappa_overlap_rows")}, indent=2))


if __name__ == "__main__":
    main()