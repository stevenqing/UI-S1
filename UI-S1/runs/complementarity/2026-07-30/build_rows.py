import argparse
import ast
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO

from scoring import (
    ACTION_TO_ID,
    GROUNDING_ACTIONS,
    distance_to_bbox,
    label_android_row,
    normalized_bbox,
    read_jsonl,
    score_mind2web_row,
    token_f1,
)


AC_MODELS = ("ui-agile-3b", "ui-agile-7b", "ui-r1-e-3b", "gui-r1-3b", "gui-r1-7b")
M2W_SPECS = {
    "showui-2b": ("showui", "runs/mind2web-showui/2026-07-28/artifacts/merged/predictions.jsonl"),
    "cogagent-18b": ("cogagent", "runs/mind2web-cogagent/2026-07-28/artifacts/merged/predictions.jsonl"),
    "qwen2.5-vl-3b": ("tongui", "runs/mind2web-tongui/2026-07-28/artifacts/qwen-3b/merged/predictions.jsonl"),
    "qwen2.5-vl-7b": ("tongui", "runs/mind2web-tongui/2026-07-28/artifacts/qwen-7b/merged/predictions.jsonl"),
    "tongui-3b": ("tongui", "runs/mind2web-tongui/2026-07-28/artifacts/tongui-3b/merged/predictions.jsonl"),
    "tongui-7b": ("tongui", "runs/mind2web-tongui/2026-07-28/artifacts/tongui-7b/merged/predictions.jsonl"),
    "tongui-32b": ("tongui", "runs/mind2web-tongui/2026-07-28/artifacts/tongui-32b/full/predictions.jsonl"),
    "ui-tars-2b": ("uitars", "runs/mind2web-uitars/2026-07-28/artifacts/2b/merged/predictions.jsonl"),
    "ui-tars-7b": ("uitars", "runs/mind2web-uitars/2026-07-28/artifacts/7b/merged/predictions.jsonl"),
    "ui-tars-72b": ("uitars", "runs/mind2web-uitars/2026-07-28/artifacts/72b/full/predictions.jsonl"),
}
EXPECTED_INPUT_SHA256 = {
    "androidcontrol_summary.json": "5c4e9495c1b1eaaee46fad7101ef148174bb41d1cf51c5b99b4931ff2188adfb",
    "mind2web_summary.json": "f0418b3f42806ac6026cadc7e35ad4c948128256c20294798e1e3ff21baa6609",
    "seeclick_native.jsonl": "00bb6d7f53725b19be71e710b9b92044dca502936851dabd5275a9949aec035f",
    "seeclick_supplement.jsonl": "832e87d9195607dd17048029c05560961bfa11a840224f0429a8a6af6b55b8c3",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_features(image_bytes=None, image_path: Path | None = None) -> dict:
    image = Image.open(BytesIO(image_bytes) if image_bytes is not None else image_path).convert("RGB")
    width, height = image.size
    thumbnail = image.copy()
    thumbnail.thumbnail((128, 128))
    values = np.asarray(thumbnail, dtype=np.float32)
    gray = values.mean(axis=2)
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    horizontal = np.abs(np.diff(gray, axis=1)) if gray.shape[1] > 1 else np.zeros((1, 1))
    vertical = np.abs(np.diff(gray, axis=0)) if gray.shape[0] > 1 else np.zeros((1, 1))
    edge_density = float((np.count_nonzero(horizontal > 20) + np.count_nonzero(vertical > 20)) /
                         (horizontal.size + vertical.size))
    return {
        "image_width": width,
        "image_height": height,
        "image_gray_mean": float(gray.mean() / 255),
        "image_gray_std": float(gray.std() / 255),
        "image_entropy": entropy,
        "image_edge_density": edge_density,
    }


def ac_cross_setting_map(low_rows: list[dict], high_rows: list[dict]) -> tuple[dict[int, int], set[int], set[int]]:
    def key(row):
        return row["image_sha256"], row["gt_action"], tuple(row["gt_bbox"])

    high_groups = defaultdict(list)
    for index, row in enumerate(high_rows):
        high_groups[key(row)].append(index)
    mapping = {}
    conflict_low = set()
    conflict_high = set()
    low_groups = defaultdict(list)
    for index, row in enumerate(low_rows):
        low_groups[key(row)].append(index)
    if {k: len(v) for k, v in low_groups.items()} != {k: len(v) for k, v in high_groups.items()}:
        raise ValueError("AndroidControl Low/High identity multisets differ")
    for identity_key, low_indices in low_groups.items():
        by_parameter = defaultdict(deque)
        for high_index in high_groups[identity_key]:
            by_parameter[high_rows[high_index]["gt_input_text"]].append(high_index)
        unmatched_low = []
        for low_index in low_indices:
            queue = by_parameter[low_rows[low_index]["gt_input_text"]]
            if queue:
                mapping[low_index] = queue.popleft()
            else:
                unmatched_low.append(low_index)
        unmatched_high = sorted(index for queue in by_parameter.values() for index in queue)
        if len(unmatched_low) != len(unmatched_high):
            raise ValueError("AndroidControl duplicate-aware matching failed")
        for low_index, high_index in zip(sorted(unmatched_low), unmatched_high):
            mapping[low_index] = high_index
            conflict_low.add(low_index)
            conflict_high.add(high_index)
    if len(mapping) != 7708 or len(conflict_low) != 58:
        raise ValueError("expected 7,708 AndroidControl pairs with 58 parameter conflicts")
    return mapping, conflict_low, conflict_high


def ac_source_features(root: Path, low_rows: list[dict]) -> dict[str, dict]:
    data_path = root / "runs/androidcontrol-rft/2026-07-29/data/UI-AGILE-Data/android_control/androidcontrol_low_test.parquet"
    images = pq.read_table(data_path, columns=["image"]).column("image").to_pylist()
    if len(images) != len(low_rows):
        raise ValueError("AndroidControl image/source count mismatch")
    cache = {}
    for row, image in zip(low_rows, images):
        digest = hashlib.sha256(image["bytes"]).hexdigest()
        if digest != row["image_sha256"]:
            raise ValueError(f"AndroidControl image hash mismatch at {row['index']}")
        if digest not in cache:
            cache[digest] = image_features(image_bytes=image["bytes"])
    return cache


def base_record(**values) -> dict:
    nan = float("nan")
    defaults = {
        "bench": "", "setting": "", "row_id": "", "episode_id": "", "group_key": "",
        "group_source": "", "model": "", "instruction": "", "history": "", "history_len": 0,
        "image_ref": "", "image_width": 0, "image_height": 0, "image_gray_mean": nan,
        "image_gray_std": nan, "image_entropy": nan, "image_edge_density": nan,
        "gt_action": "", "gt_x": nan, "gt_y": nan, "gt_bbox": None, "gt_param": "",
        "pred_action": "", "pred_x": nan, "pred_y": nan, "pred_param": "", "pred_raw": "",
        "parse_ok": False, "err_label": "", "success": False, "ground_dist": nan,
        "bbox_dist": nan, "quarantine": False,
    }
    defaults.update(values)
    return defaults


def build_androidcontrol(root: Path, summary: dict) -> tuple[list[dict], dict]:
    artifacts = root / "runs/androidcontrol-rft/2026-07-29/artifacts"
    source_by_setting = {
        setting: read_jsonl(artifacts / "ui-agile-3b" / setting / "predictions.jsonl")
        for setting in ("low", "high")
    }
    low_to_high, conflict_low, conflict_high = ac_cross_setting_map(
        source_by_setting["low"], source_by_setting["high"]
    )
    high_episode = {}
    for high_index, row in enumerate(source_by_setting["high"]):
        goal = row["instruction"].strip()
        high_episode[high_index] = "ac_goal_" + sha256_text(goal)[:20]
    low_episode = {low_index: high_episode[high_index] for low_index, high_index in low_to_high.items()}
    visual = ac_source_features(root, source_by_setting["low"])
    records = []
    lane_audit = {}
    for setting in ("low", "high"):
        reference = source_by_setting[setting]
        conflict_indices = conflict_low if setting == "low" else conflict_high
        for model in AC_MODELS:
            rows = read_jsonl(artifacts / model / setting / "predictions.jsonl")
            if len(rows) != 7708 or [row["index"] for row in rows] != list(range(7708)):
                raise ValueError(f"incomplete AndroidControl lane: {model}/{setting}")
            labels = [label_android_row(row) for row in rows]
            expected = summary["models"][model][setting]
            if sum(label["step_success"] for label in labels) != expected["metric_counts"]["step_success"]["correct"]:
                raise ValueError(f"AndroidControl Step SR mismatch: {model}/{setting}")
            if dict(sorted(Counter(label["error_type"] for label in labels).items())) != expected["error_type_counts"]:
                raise ValueError(f"AndroidControl error-label mismatch: {model}/{setting}")
            for index, (row, label) in enumerate(zip(rows, labels)):
                source = reference[index]
                if row["source_sha256"] != source["source_sha256"]:
                    raise ValueError(f"AndroidControl source mismatch: {model}/{setting}/{index}")
                width, height = row["image_size"]
                pred_position = row["pred_coord"][:2] if row["pred_coord"] else None
                has_gt_point = row["gt_action"] in GROUNDING_ACTIONS
                episode_id = low_episode[index] if setting == "low" else high_episode[index]
                history = row.get("history") or ""
                record = base_record(
                    bench="androidcontrol", setting=setting, row_id=str(index), episode_id=episode_id,
                    group_key=episode_id, group_source="high_full_instruction_hash",
                    model=model, instruction=row["instruction"], history=history,
                    history_len=history.count("Step "),
                    image_ref=f"androidcontrol:{row['image_sha256']}",
                    gt_action=row["gt_action"],
                    gt_x=row["gt_bbox"][0] / width if has_gt_point else float("nan"),
                    gt_y=row["gt_bbox"][1] / height if has_gt_point else float("nan"),
                    gt_param=row["gt_input_text"], pred_action=row["pred_action"] or "",
                    pred_x=pred_position[0] / width if pred_position else float("nan"),
                    pred_y=pred_position[1] / height if pred_position else float("nan"),
                    pred_param=row["pred_input_text"] if isinstance(row["pred_input_text"], str) else "",
                    pred_raw=row["pred_raw"], parse_ok=row["pred_action"] is not None,
                    err_label=label["error_type"], success=label["step_success"],
                    ground_dist=label["normalized_distance"] if label["normalized_distance"] is not None else float("nan"),
                    quarantine=index in conflict_indices,
                    **visual[row["image_sha256"]],
                )
                records.append(record)
            lane_audit[f"{setting}/{model}"] = {
                "rows": len(rows), "successes": sum(label["step_success"] for label in labels),
                "quarantine_rows": len(conflict_indices),
            }
    return records, {
        "lanes": lane_audit,
        "episode_groups": len(set(high_episode.values())),
        "group_source": "SHA256 of released High full instruction; identical goals are conservatively merged",
        "parameter_conflict_pairs": len(conflict_low),
    }


def m2w_metadata(root: Path) -> tuple[dict, list[dict]]:
    episodes = json.loads((root / "runs/mind2web/2026-07-27/data/mind2web_data_test_task.json").read_text())
    by_identity = {}
    actions = []
    for episode in episodes:
        for step_index, action in enumerate(episode["actions"]):
            key = (episode["annotation_id"], action["action_uid"])
            value = {
                "episode": episode,
                "action": action,
                "step_index": step_index,
                "history": "\n".join(episode["action_reprs"][:step_index]),
            }
            by_identity[key] = value
            actions.append(value)
    if len(by_identity) != 2094:
        raise ValueError("expected 2,094 Mind2Web source actions")
    return by_identity, actions


def seeclick_rows(root: Path) -> dict[tuple[str, str], dict]:
    output = {}
    paths = (
        (root / "runs/mind2web/2026-07-27/artifacts/gate1_cross_task/predictions.jsonl", "seeclick_native.jsonl"),
        (root / "runs/mind2web/2026-07-27/artifacts/gate1_corrected_missing/predictions.jsonl", "seeclick_supplement.jsonl"),
    )
    for path, hash_key in paths:
        actual_hash = sha256_file(path)
        if actual_hash != EXPECTED_INPUT_SHA256[hash_key]:
            raise ValueError(f"SeeClick source hash mismatch: {path} {actual_hash}")
        for row in read_jsonl(path):
            stem = Path(row["img_path"]).stem
            annot_id = "-".join(stem.split("-")[:5])
            key = (annot_id, stem[len(annot_id) + 1:])
            if key in output:
                raise ValueError(f"duplicate SeeClick identity across native/supplement: {key}")
            output[key] = row
    if len(output) != 2080:
        raise ValueError("expected 2,080 SeeClick rows")
    return output


def build_mind2web(root: Path, summary: dict) -> tuple[list[dict], dict]:
    parsers = {
        "showui": load_module(root / "runs/mind2web-showui/2026-07-28/score.py", "build_showui_score").parse_prediction,
        "tongui": load_module(root / "runs/mind2web-tongui/2026-07-28/score.py", "build_tongui_score").parse_prediction,
        "cogagent": load_module(root / "runs/mind2web-cogagent/2026-07-28/common.py", "build_cogagent_common").parse_prediction,
        "uitars": load_module(root / "runs/mind2web-uitars/2026-07-28/common.py", "build_uitars_common").parse_prediction,
    }
    reference = read_jsonl(root / M2W_SPECS["showui-2b"][1])
    identities = [(row["annot_id"], row["action_uid"]) for row in reference]
    if len(reference) != 2080 or len(set(identities)) != 2080:
        raise ValueError("Mind2Web visual reference is incomplete")
    metadata, all_actions = m2w_metadata(root)
    image_root = root / "runs/mind2web/2026-07-27/data/ming2web_images"
    visual_features = {}
    for row in reference:
        image_path = image_root / row["image"]
        visual_features[(row["annot_id"], row["action_uid"])] = image_features(image_path=image_path)

    records = []
    lane_audit = {}
    for model, (kind, path) in M2W_SPECS.items():
        rows = read_jsonl(root / path)
        if [(row["annot_id"], row["action_uid"]) for row in rows] != identities:
            raise ValueError(f"Mind2Web identity mismatch: {model}")
        labels = [score_mind2web_row(row, kind, parsers[kind]) for row in rows]
        expected = summary["models"][model]
        if sum(label["step_success"] for label in labels) != round(expected["metrics"]["step_success_micro"] * 2080):
            raise ValueError(f"Mind2Web Step SR mismatch: {model}")
        if dict(sorted(Counter(label["error_type"] for label in labels).items())) != expected["error_type_counts"]:
            raise ValueError(f"Mind2Web error-label mismatch: {model}")
        for row, label in zip(rows, labels):
            key = (row["annot_id"], row["action_uid"])
            meta = metadata[key]
            episode = meta["episode"]
            bbox = normalized_bbox(row, rounded=False)
            position = label["position"]
            gt_position = row["answer"]["position"]
            record = base_record(
                bench="mind2web", setting="visual", row_id=f"{key[0]}__{key[1]}", episode_id=key[0],
                group_key=episode.get("website") or episode.get("domain") or key[0],
                group_source="website_fallback_domain", model=model,
                instruction=episode["confirmed_task"], history=meta["history"], history_len=meta["step_index"],
                image_ref=str((image_root / row["image"]).relative_to(root)),
                gt_action=row["answer"]["action"], gt_x=gt_position[0], gt_y=gt_position[1],
                gt_bbox=bbox, gt_param=row["answer"].get("value") or "",
                pred_action=label["pred_action"] or "",
                pred_x=position[0] if position else float("nan"), pred_y=position[1] if position else float("nan"),
                pred_param=label.get("pred_param") or "", pred_raw=row["response"],
                parse_ok=label["parse_ok"], err_label=label["error_type"], success=label["step_success"],
                ground_dist=math.dist(position, gt_position) if position else float("nan"),
                bbox_dist=label.get("bbox_distance") if label.get("bbox_distance") is not None else float("nan"),
                **visual_features[key],
            )
            records.append(record)
        lane_audit[f"visual/{model}"] = {"rows": 2080, "successes": sum(x["step_success"] for x in labels)}

    see_rows = seeclick_rows(root)
    see_labels = []
    for reference_row in reference:
        key = (reference_row["annot_id"], reference_row["action_uid"])
        row = see_rows[key]
        try:
            prediction = ast.literal_eval(row["sentence"])
            pred_action = {4: "CLICK", 2: "SELECT", 3: "TYPE"}.get(prediction["action_type"])
            position = list(prediction.get("click_point")) if prediction.get("click_point") is not None else None
            pred_param = prediction.get("value") or ""
        except (KeyError, SyntaxError, TypeError, ValueError):
            pred_action, position, pred_param = None, None, ""
        operation_f1 = float(row["Op_F1"][0])
        element = float(row["Ele_match"])
        success = bool(element == 1.0 and operation_f1 == 1.0)
        if success:
            error_type = "success"
        elif not row["parse_ok"]:
            error_type = "parse_failure"
        elif pred_action != reference_row["answer"]["action"]:
            error_type = "action_mismatch"
        elif operation_f1 != 1.0:
            error_type = "parameter_miss"
        else:
            error_type = "element_miss"
        see_labels.append((row, pred_action, position, pred_param, success, error_type))
    expected_see = summary["models"]["seeclick-9.6b"]
    if sum(item[4] for item in see_labels) != round(expected_see["metrics"]["step_success_micro"] * 2080):
        raise ValueError("SeeClick Step SR mismatch")
    if dict(sorted(Counter(item[5] for item in see_labels).items())) != expected_see["error_type_counts"]:
        raise ValueError("SeeClick error-label mismatch")
    for reference_row, item in zip(reference, see_labels):
        row, pred_action, position, pred_param, success, error_type = item
        key = (reference_row["annot_id"], reference_row["action_uid"])
        meta = metadata[key]
        episode = meta["episode"]
        bbox = normalized_bbox(reference_row, rounded=True)
        gt_position = reference_row["answer"]["position"]
        records.append(base_record(
            bench="mind2web", setting="visual", row_id=f"{key[0]}__{key[1]}", episode_id=key[0],
            group_key=episode.get("website") or episode.get("domain") or key[0], group_source="website_fallback_domain",
            model="seeclick-9.6b", instruction=episode["confirmed_task"], history=meta["history"],
            history_len=meta["step_index"], image_ref=str((image_root / reference_row["image"]).relative_to(root)),
            gt_action=reference_row["answer"]["action"], gt_x=gt_position[0], gt_y=gt_position[1],
            gt_bbox=normalized_bbox(reference_row, rounded=False), gt_param=reference_row["answer"].get("value") or "",
            pred_action=pred_action or "", pred_x=position[0] if position else float("nan"),
            pred_y=position[1] if position else float("nan"), pred_param=pred_param, pred_raw=row["sentence"],
            parse_ok=bool(row["parse_ok"]), err_label=error_type, success=success,
            ground_dist=math.dist(position, gt_position) if position else float("nan"),
            bbox_dist=distance_to_bbox(position, bbox) if position else float("nan"), **visual_features[key],
        ))
    lane_audit["visual/seeclick-9.6b"] = {"rows": 2080, "successes": sum(item[4] for item in see_labels)}

    predictions = json.loads((root / "runs/mindact/2026-07-29/artifacts/full/test_task_predictions_top50.json").read_text())
    prediction_by_identity = {tuple(item[0].rsplit("_", 1)): item for item in predictions}
    if len(prediction_by_identity) != 2094:
        raise ValueError("MindAct prediction identity set is incomplete")
    mindact_labels = []
    reference_by_identity = {key: row for key, row in zip(identities, reference)}
    for meta in all_actions:
        episode, action = meta["episode"], meta["action"]
        key = (episode["annotation_id"], action["action_uid"])
        prediction = prediction_by_identity[key]
        positives = {candidate["backend_node_id"] for candidate in action["pos_candidates"] if candidate.get("rank", 10**9) < 50}
        element = float(prediction[1] in positives)
        reference_action = action["operation"]["op"]
        reference_operation = reference_action
        if reference_action != "CLICK":
            reference_operation += " " + action["operation"]["value"]
        operation_f1 = token_f1(prediction[2], reference_operation)
        success = bool(element == 1.0 and operation_f1 == 1.0)
        pieces = prediction[2].strip().split(maxsplit=1)
        pred_action = pieces[0] if pieces else ""
        pred_param = pieces[1] if len(pieces) == 2 else ""
        error_type = (
            "success" if success else "action_mismatch" if pred_action != reference_action
            else "parameter_miss" if operation_f1 != 1.0 else "element_miss"
        )
        visual_row = reference_by_identity.get(key)
        gt_bbox = normalized_bbox(visual_row, rounded=False) if visual_row else None
        gt_position = visual_row["answer"]["position"] if visual_row else None
        feature_values = visual_features[key] if key in visual_features else {}
        mindact_labels.append((success, error_type))
        records.append(base_record(
            bench="mind2web", setting="html", row_id=f"{key[0]}__{key[1]}", episode_id=key[0],
            group_key=episode.get("website") or episode.get("domain") or key[0], group_source="website_fallback_domain",
            model="mindact-flan-t5-xl", instruction=episode["confirmed_task"], history=meta["history"],
            history_len=meta["step_index"], image_ref=str((image_root / visual_row["image"]).relative_to(root)) if visual_row else "",
            gt_action=reference_action, gt_x=gt_position[0] if gt_position else float("nan"),
            gt_y=gt_position[1] if gt_position else float("nan"), gt_bbox=gt_bbox,
            gt_param=action["operation"].get("value") or "", pred_action=pred_action,
            pred_param=pred_param, pred_raw=prediction[2], parse_ok=bool(pieces),
            err_label=error_type, success=success, **feature_values,
        ))
    expected_mindact = summary["mindact_html"]
    if sum(item[0] for item in mindact_labels) != round(expected_mindact["full_metrics"]["step_acc"] * 2094):
        raise ValueError("MindAct Step SR mismatch")
    lane_audit["html/mindact-flan-t5-xl"] = {"rows": 2094, "successes": sum(item[0] for item in mindact_labels)}
    return records, {"lanes": lane_audit, "episodes": 252, "website_groups": len({e["website"] for e in (m["episode"] for m in all_actions)})}


def assign_group_folds(records: list[dict], n_splits: int = 5) -> dict:
    rows_by_pool = defaultdict(set)
    for row in records:
        if row["quarantine"]:
            continue
        pool = f"{row['bench']}/{row['setting']}"
        rows_by_pool[pool].add((row["row_id"], row["group_key"]))
    output = {"n_splits": n_splits, "method": "deterministic_group_balance_v1", "pools": {}}
    for pool, identities in sorted(rows_by_pool.items()):
        group_counts = Counter(group for _, group in identities)
        fold_load = [0] * n_splits
        group_to_fold = {}
        for group, count in sorted(group_counts.items(), key=lambda item: (-item[1], sha256_text(item[0]))):
            fold = min(range(n_splits), key=lambda index: (fold_load[index], index))
            group_to_fold[group] = fold
            fold_load[fold] += count
        output["pools"][pool] = {
            "groups": len(group_counts), "rows": len(identities), "fold_rows": fold_load,
            "group_to_fold": group_to_fold,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    upstream = root / "runs/error-overlap-analysis/2026-07-29"
    ac_summary_path = upstream / "androidcontrol_summary.json"
    m2w_summary_path = upstream / "mind2web_summary.json"
    for path in (ac_summary_path, m2w_summary_path):
        actual_hash = sha256_file(path)
        if actual_hash != EXPECTED_INPUT_SHA256[path.name]:
            raise ValueError(f"upstream summary hash mismatch: {path} {actual_hash}")
    ac_summary = json.loads(ac_summary_path.read_text())
    m2w_summary = json.loads(m2w_summary_path.read_text())
    if ac_summary["status"] != "PASS" or m2w_summary["status"] != "PASS":
        raise ValueError("upstream overlap summaries are not PASS")

    ac_records, ac_audit = build_androidcontrol(root, ac_summary)
    m2w_records, m2w_audit = build_mind2web(root, m2w_summary)
    records = ac_records + m2w_records
    if len(records) != 102054:
        raise ValueError(f"expected 102,054 tidy rows, found {len(records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, args.output, compression="zstd", use_dictionary=True)
    folds = assign_group_folds(records)
    args.folds.write_text(json.dumps(folds, indent=2, sort_keys=True) + "\n")
    manifest = {
        "status": "PASS",
        "rows": len(records),
        "row_counts": dict(sorted(Counter(f"{r['bench']}/{r['setting']}" for r in records).items())),
        "quarantine_tidy_rows": sum(r["quarantine"] for r in records),
        "quarantine_semantics": "58 paired identities x 2 settings x 5 AndroidControl models",
        "default_filter": "quarantine == false",
        "label_provenance": "shared scoring.py; per-lane Step SR and exclusive counts matched upstream summaries",
        "locked_input_sha256": EXPECTED_INPUT_SHA256,
        "androidcontrol": ac_audit,
        "mind2web": m2w_audit,
        "schema": str(table.schema),
        "rows_parquet_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "folds_sha256": hashlib.sha256(args.folds.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "rows", "row_counts", "quarantine_tidy_rows")}, indent=2))


if __name__ == "__main__":
    main()