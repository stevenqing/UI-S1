import collections
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
R1_CONFIG_PATH = RUN_DIR / "configs/r1_headroom.yaml"
RECOVERY_MANIFEST_PATH = RUN_DIR / "RECOVERY_MANIFEST.json"
sys.path.insert(0, str(RUN_DIR))

from recovery_common import (
    assert_protected_process, atomic_json, identity_hash, load_config, load_jsonl,
    references, sha256_file, validate_lane_rows,
)


def load_locked_recovery(path=RECOVERY_MANIFEST_PATH):
    path = Path(path)
    if not path.is_file():
        raise PermissionError("TriVUS R1 sealed until R0 recovery manifest exists")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("status") != "PASS_TRIVUS_R0_RESULT_BLIND_RECOVERY"
        or manifest.get("ground_truth_fields_used") is not False
        or manifest.get("scorer_or_evaluator_imported") is not False
        or manifest.get("accuracy_or_oracle_computed") is not False
        or manifest.get("historical_files_modified") is not False
        or manifest.get("expected_rows_per_lane") != 2000
        or len(manifest.get("lanes", {})) != 6
        or any(lane.get("rows") != 2000 for lane in manifest.get("lanes", {}).values())
    ):
        raise PermissionError("TriVUS R1 invalid R0 recovery manifest")
    recovery_config = load_config()
    assert_protected_process(recovery_config)
    for name, lane in recovery_config["lanes"].items():
        record = manifest["lanes"][name]
        path = ROOT / lane["destination"]
        rows = validate_lane_rows(
            load_jsonl(path), references(recovery_config, lane["setting"]),
            lane, require_complete=True,
        )
        expected = {
            "model_id": lane["model_id"],
            "setting": lane["setting"],
            "path": lane["destination"],
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "row_ids_sha256": identity_hash(rows),
            "recovered_from_rows": lane["seed_rows"],
        }
        if record != expected:
            raise PermissionError(f"TriVUS R1 recovered-lane hash mismatch: {name}")
    for name, lane in recovery_config["complete_lanes"].items():
        record = manifest["lanes"][name]
        rows = []
        files = []
        for item in lane["shards"]:
            path = ROOT / item["path"]
            shard_rows = load_jsonl(path)
            if (
                len(shard_rows) != item["rows"]
                or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]
            ):
                raise PermissionError(f"TriVUS R1 complete-shard mismatch: {name}/{item['shard_index']}")
            rows.extend(shard_rows)
            files.append({
                "path": item["path"],
                "rows": item["rows"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "shard_index": item["shard_index"],
            })
        rows = validate_lane_rows(
            rows, references(recovery_config, lane["setting"]),
            lane, require_complete=True,
        )
        expected = {
            "model_id": lane["model_id"],
            "setting": lane["setting"],
            "rows": len(rows),
            "row_ids_sha256": identity_hash(rows),
            "files": files,
            "preexisting_complete": True,
        }
        if record != expected:
            raise PermissionError(f"TriVUS R1 complete-lane manifest mismatch: {name}")
    return recovery_config, manifest


def load_r1_config():
    config = yaml.safe_load(R1_CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_BEFORE_R0_COMPLETION_AND_R1_RESULTS":
        raise ValueError("TriVUS R1 protocol is not frozen")
    if config.get("models") != ["UI-AGILE-7B", "GUI-R1-7B", "UI-R1-E-3B"]:
        raise ValueError("TriVUS R1 model order mismatch")
    if config.get("settings") != ["low", "high"] or config.get("expected_rows_per_setting") != 2000:
        raise ValueError("TriVUS R1 setting contract mismatch")
    if config.get("practical_headroom_margin") != 0.01:
        raise ValueError("TriVUS R1 margin mismatch")
    if config.get("bootstrap", {}).get("resamples") != 10000 or config["bootstrap"].get("confidence") != 0.99:
        raise ValueError("TriVUS R1 bootstrap contract mismatch")
    for item in config["dependencies"].values():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"TriVUS R1 dependency hash mismatch: {item['path']}")
    return config


def load_scoring_and_bootstrap(config):
    collision = ROOT / "runs/collision-law/2026-07-30"
    sys.path.insert(0, str(collision))
    from scoring import GROUNDING_ACTIONS, SIMPLE_ACTIONS, TEXT_ACTIONS, text_f1

    bootstrap_path = ROOT / config["dependencies"]["bootstrap"]["path"]
    spec = importlib.util.spec_from_file_location("trivus_r1_bootstrap", bootstrap_path)
    if spec is None or spec.loader is None:
        raise ImportError(bootstrap_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return GROUNDING_ACTIONS, SIMPLE_ACTIONS, TEXT_ACTIONS, text_f1, module.paired_bootstrap


def load_lane_maps(recovery_config):
    output = {setting: {} for setting in ("low", "high")}
    for lane in recovery_config["lanes"].values():
        rows = load_jsonl(ROOT / lane["destination"])
        output[lane["setting"]][lane["model_id"]] = {row["id"]: row for row in rows}
    for lane in recovery_config["complete_lanes"].values():
        rows = []
        for item in lane["shards"]:
            rows.extend(load_jsonl(ROOT / item["path"]))
        output[lane["setting"]][lane["model_id"]] = {row["id"]: row for row in rows}
    return output


def reference_value(row):
    width, height = row["image_size"]
    coordinate = row["gt_bbox"]
    parameter = "" if row["gt_input_text"] == "no input text" else row["gt_input_text"]
    return {
        "action": row["gt_action"],
        "x": coordinate[0] / width if coordinate[0] >= 0 else None,
        "y": coordinate[1] / height if coordinate[1] >= 0 else None,
        "parameter": parameter,
        "group": row["episode_id"],
    }


def candidate_value(row):
    prediction = row["prediction"]
    coordinate = prediction["position"]
    return {
        "action": str(prediction.get("action") or ""),
        "x": float(coordinate[0]) if coordinate is not None else None,
        "y": float(coordinate[1]) if coordinate is not None else None,
        "parameter": str(prediction.get("value") or ""),
        "source": row["model_id"],
        "parse_ok": bool(prediction.get("parse_ok")),
    }


def score_candidate(reference, candidate, scoring, coordinate_radius, text_threshold):
    grounding_actions, simple_actions, text_actions, text_f1 = scoring
    if not candidate["parse_ok"] or candidate["action"] != reference["action"]:
        return False
    if reference["action"] in grounding_actions:
        if candidate["x"] is None or candidate["y"] is None:
            return False
        return math.dist((candidate["x"], candidate["y"]), (reference["x"], reference["y"])) < coordinate_radius
    if reference["action"] in text_actions:
        return text_f1(candidate["parameter"], reference["parameter"]) >= text_threshold
    if reference["action"] in simple_actions:
        return True
    raise ValueError(f"unknown AndroidControl action: {reference['action']}")


def majority_candidate(candidates, priority):
    parsed = [candidate for candidate in candidates if candidate["parse_ok"]]
    if not parsed:
        return None
    counts = collections.Counter(candidate["action"] for candidate in parsed)
    highest = max(counts.values())
    tied = {action for action, count in counts.items() if count == highest}
    return next(
        candidate for source in priority for candidate in parsed
        if candidate["source"] == source and candidate["action"] in tied
    )


def gate_pass(report, margin):
    return report["point_delta"] > margin and report["ci_99"][0] > 0


def analyze_setting(setting, config, recovery_config, lane_maps, scoring, paired_bootstrap):
    reference_rows = load_jsonl(ROOT / recovery_config["references"][setting]["path"])
    references = {row["id"]: reference_value(row) for row in reference_rows}
    row_ids = [row["id"] for row in reference_rows]
    models = config["models"]
    if any(set(lane_maps[setting][model]) != set(row_ids) for model in models):
        raise ValueError(f"TriVUS R1 lane/reference identity mismatch: {setting}")
    folds = json.loads((ROOT / config["dependencies"]["folds"]["path"]).read_text())
    fold_map = folds["pools"][f"androidcontrol/{setting}"]["group_to_fold"]
    candidates = {
        row_id: [candidate_value(lane_maps[setting][model][row_id]) for model in models]
        for row_id in row_ids
    }
    success = {
        row_id: {
            candidate["source"]: score_candidate(
                references[row_id], candidate, scoring,
                config["coordinate_radius"], config["text_f1_threshold"],
            )
            for candidate in candidates[row_id]
        }
        for row_id in row_ids
    }
    outputs = {"majority": {}, "oracle": {}, **{model: {} for model in models}}
    fold_reports = []
    for test_fold in range(5):
        dev_ids = [row_id for row_id in row_ids if fold_map[references[row_id]["group"]] != test_fold]
        test_ids = [row_id for row_id in row_ids if fold_map[references[row_id]["group"]] == test_fold]
        reliability = {
            model: float(np.mean([success[row_id][model] for row_id in dev_ids]))
            for model in models
        }
        priority = sorted(models, key=lambda model: (-reliability[model], models.index(model)))
        for row_id in test_ids:
            majority = majority_candidate(candidates[row_id], priority)
            outputs["majority"][row_id] = False if majority is None else success[row_id][majority["source"]]
            outputs["oracle"][row_id] = any(success[row_id].values())
            for model in models:
                outputs[model][row_id] = success[row_id][model]
        fold_reports.append({
            "fold": test_fold,
            "dev_rows": len(dev_ids),
            "test_rows": len(test_ids),
            "reliability": reliability,
            "priority": priority,
        })
    if any(set(values) != set(row_ids) for values in outputs.values()):
        raise ValueError(f"TriVUS R1 output coverage mismatch: {setting}")
    metadata = {
        row_id: {"fold": fold_map[references[row_id]["group"]], "group": references[row_id]["group"]}
        for row_id in row_ids
    }
    differences = {
        row_id: int(outputs["oracle"][row_id]) - int(outputs["majority"][row_id])
        for row_id in row_ids
    }
    comparison = paired_bootstrap(
        metadata, differences, config["bootstrap"]["resamples"], config["bootstrap"]["seed"][setting]
    )
    return {
        "rows": len(row_ids),
        "groups": len(set(reference["group"] for reference in references.values())),
        "accuracy": {name: float(np.mean(list(values.values()))) for name, values in outputs.items()},
        "oracle_minus_majority": comparison,
        "folds": fold_reports,
        "outputs": outputs,
        "gate_pass": gate_pass(comparison, config["practical_headroom_margin"]),
    }


def main():
    recovery_config, recovery_manifest = load_locked_recovery()
    config = load_r1_config()
    grounding_actions, simple_actions, text_actions, text_f1, paired_bootstrap = load_scoring_and_bootstrap(config)
    scoring = (grounding_actions, simple_actions, text_actions, text_f1)
    lane_maps = load_lane_maps(recovery_config)
    settings = {
        setting: analyze_setting(
            setting, config, recovery_config, lane_maps, scoring, paired_bootstrap
        )
        for setting in config["settings"]
    }
    gates = {f"R1_{setting}_headroom": settings[setting]["gate_pass"] for setting in config["settings"]}
    result = {
        "schema_version": 1,
        "status": "PASS_TRIVUS_R1_ADJUDICATED",
        "outcome": "PROCEED_TO_BLIND_SELECTOR" if all(gates.values()) else "STOP_TRIVUS_T_K2",
        "gates": gates,
        "settings": settings,
        "recovery_manifest_sha256": sha256_file(RECOVERY_MANIFEST_PATH),
        "config_sha256": sha256_file(R1_CONFIG_PATH),
    }
    output = RUN_DIR / "R1_HEADROOM.json"
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, result)
    print(json.dumps({
        "outcome": result["outcome"],
        "gates": gates,
        "settings": {
            setting: {
                "accuracy": value["accuracy"],
                "oracle_minus_majority": value["oracle_minus_majority"],
            }
            for setting, value in settings.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()