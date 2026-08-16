import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/icc_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
STAGE0_PATH = ROOT / "runs/evid/2026-08-15/STAGE0.json"
STAGE1_PATH = ROOT / "runs/evid/2026-08-15/STAGE1.json"
OUTPUT_PATH = RUN_DIR / "ARM_B.json"
RAW_PATH = RUN_DIR / "raw/arm_b_changed_rows.jsonl"

MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def lineage_counts(block):
    counts = [0, 0, 0]
    for index in block:
        counts[index % 3] += 1
    return tuple(counts)


def direction_class(old_block, new_block):
    old = lineage_counts(old_block)
    new = lineage_counts(new_block)
    old_l = sum(value > 0 for value in old)
    new_l = sum(value > 0 for value in new)
    if new_l > old_l:
        return "diversity_increase"
    if new_l < old_l:
        return "diversity_decrease"
    if max(new) > max(old):
        return "same_L_concentration_increase"
    if max(new) < max(old):
        return "same_L_concentration_decrease"
    if new != old:
        return "lineage_substitution"
    return "composition_same"


def outcome_class(changed, before, after):
    if not changed:
        return "unchanged_correct" if before else "unchanged_wrong"
    if not before and after:
        return "wrong_to_correct"
    if before and not after:
        return "correct_to_wrong"
    if before and after:
        return "correct_to_correct"
    return "wrong_to_wrong"


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main():
    if OUTPUT_PATH.exists() or RAW_PATH.exists():
        raise FileExistsError("ICC Arm B output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    stage0 = json.loads(STAGE0_PATH.read_text())
    stage1 = json.loads(STAGE1_PATH.read_text())
    if stage0["status"] != "PASS_EVID_STAGE0_COMPLETE" or stage1["status"] != "PASS_EVID_STAGE1_COMPLETE":
        raise PermissionError("ICC Arm B EVID source mismatch")
    stage0_rows = {row["row_id"]: row for row in read_jsonl(ROOT / stage0["raw"]["path"])}
    stage1_rows = {row["row_id"]: row for row in read_jsonl(ROOT / stage1["raw"]["path"])}
    if set(stage0_rows) != set(stage1_rows) or len(stage0_rows) != 1581:
        raise ValueError("ICC Arm B row identity mismatch")
    outcome_counts = Counter()
    changed_rows = []
    direction_values = defaultdict(list)
    for row_id in sorted(stage0_rows):
        structure = stage0_rows[row_id]
        evaluation = stage1_rows[row_id]
        changed = tuple(structure["b3_block"]) != tuple(structure["fixed_block"])
        before = bool(evaluation["baselines"]["B3"])
        after = bool(evaluation["variants"]["fixed"])
        outcome = outcome_class(changed, before, after)
        outcome_counts[outcome] += 1
        if changed:
            direction = direction_class(structure["b3_block"], structure["fixed_block"])
            direction_values[direction].append((before, after))
            changed_rows.append({
                "row_id": row_id,
                "application": structure["application"],
                "fold": structure["fold"],
                "old_block": structure["b3_block"],
                "new_block": structure["fixed_block"],
                "old_lineage_counts": list(lineage_counts(structure["b3_block"])),
                "new_lineage_counts": list(lineage_counts(structure["fixed_block"])),
                "old_representative": structure["b3_representative"],
                "new_representative": structure["fixed_representative"],
                "before_correct": before,
                "after_correct": after,
                "outcome_class": outcome,
                "direction_class": direction,
            })
    if len(changed_rows) != stage0["disagreement"]["rows"]:
        raise ValueError("ICC Arm B changed-row anchor mismatch")
    direction_summary = {}
    for direction in config["arm_b"]["direction_classes"]:
        values = direction_values[direction]
        beneficial = sum(not before and after for before, after in values)
        harmful = sum(before and not after for before, after in values)
        initially_wrong = sum(not before for before, _ in values)
        direction_summary[direction] = {
            "rows": len(values),
            "wrong_to_correct": beneficial,
            "correct_to_wrong": harmful,
            "initially_wrong_rows": initially_wrong,
            "wrong_to_correct_rate_all": float(beneficial / len(values)) if values else None,
            "wrong_to_correct_rate_given_wrong": float(beneficial / initially_wrong) if initially_wrong else None,
            "net_correct": beneficial - harmful,
        }
    write_jsonl_fsynced(RAW_PATH, changed_rows)
    beneficial = outcome_counts["wrong_to_correct"]
    harmful = outcome_counts["correct_to_wrong"]
    output = {
        "schema_version": 1,
        "status": "PASS_ICC_ARM_B_CHANGED_ROW_AUDIT",
        "rows": len(stage0_rows),
        "changed_rows": len(changed_rows),
        "changed_fraction": len(changed_rows) / len(stage0_rows),
        "outcome_counts": {name: outcome_counts[name] for name in config["arm_b"]["outcome_classes"]},
        "beneficial_changes": beneficial,
        "harmful_changes": harmful,
        "net_correct_changes": beneficial - harmful,
        "net_accuracy_delta": (beneficial - harmful) / len(stage0_rows),
        "direction_summary": direction_summary,
        "diversity_vs_concentration": {
            "diversity_increase_correction_rate": direction_summary["diversity_increase"]["wrong_to_correct_rate_all"],
            "concentration_increase_correction_rate": direction_summary["same_L_concentration_increase"]["wrong_to_correct_rate_all"],
        },
        "source_hashes": {"stage0": sha256_file(STAGE0_PATH), "stage1": sha256_file(STAGE1_PATH), "spec": sha256_file(SPEC_PATH)},
        "raw": {"path": str(RAW_PATH.relative_to(ROOT)), "rows": len(changed_rows), "bytes": RAW_PATH.stat().st_size, "sha256": sha256_file(RAW_PATH), "write_flush_fsync_per_row": True},
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "outcomes": output["outcome_counts"], "net": output["net_accuracy_delta"], "directions": direction_summary}, indent=2))


if __name__ == "__main__":
    main()