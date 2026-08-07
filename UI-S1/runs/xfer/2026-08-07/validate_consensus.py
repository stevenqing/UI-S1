import hashlib
import json
import sys
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from xfer_common import COORDINATE_ACTIONS, plurality_action


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_unique_file(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f"duplicate consensus id: {row['id']}")
        rows[row["id"]] = row
    return rows


def load_unique_dir(path):
    rows = {}
    for shard in sorted(path.glob("shard-*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate proposer id: {row['id']}")
            rows[row["id"]] = row
    return rows


def main():
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    model_order = [model["id"] for model in roster["mind2web"]["models"]]
    canonical = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    rows = load_unique_file(RUN_DIR / "raw/mind2web-consensus-roi.jsonl")
    proposer = load_unique_dir(RUN_DIR / "raw/proposer-regions")
    expected_ids = {row["id"] for row in canonical}
    if set(rows) != expected_ids or set(proposer) != expected_ids:
        raise ValueError(f"consensus coverage mismatch: rows={len(rows)}, proposer={len(proposer)}")
    triggered = 0
    fallback = 0
    for source in canonical:
        row = rows[source["id"]]
        if row["stable_index"] != source["stable_index"] or row["image_sha256"] != source["image_sha256"]:
            raise ValueError(f"consensus identity mismatch: {source['id']}")
        if any("target" in key or "bbox" in key or key == "step" for key in row):
            raise ValueError(f"consensus target leak: {source['id']}")
        candidates = row["stage1_predictions"]
        if len(candidates) != 6:
            raise ValueError(f"consensus stage1 budget mismatch: {source['id']}")
        expected_order = [(view, model) for view in (0, 1) for model in model_order]
        actual_order = [(candidate["view_index"], candidate["model"]) for candidate in candidates]
        if actual_order != expected_order:
            raise ValueError(f"consensus stage1 order mismatch: {source['id']}")
        winning_type = plurality_action(candidates, model_order)
        if row["winning_type"] != winning_type:
            raise ValueError(f"consensus winning type mismatch: {source['id']}")
        retained = [
            candidate for candidate in candidates
            if candidate.get("parse_ok") and candidate["action"] == winning_type
            and candidate.get("position") is not None
        ]
        trigger = winning_type in COORDINATE_ACTIONS and bool(retained)
        if row["stage2_trigger"] != trigger or row["trigger_candidate_count"] != len(retained):
            raise ValueError(f"consensus trigger mismatch: {source['id']}")
        arms = row["arms"]
        if set(arms) != {"C_uni", "C_cond", "C_rand", "C_self"}:
            raise ValueError(f"consensus arm set mismatch: {source['id']}")
        expected_length = 2 if trigger else 0
        if any(len(values) != expected_length for values in arms.values()):
            raise ValueError(f"consensus arm budget mismatch: {source['id']}")
        if trigger:
            expected_uni = [proposer[source["id"]]["regions"][index]["region"] for index in (1, 2)]
            if arms["C_uni"] != expected_uni:
                raise ValueError(f"consensus C-uni geometry mismatch: {source['id']}")
            triggered += 1
        if canonical_hash(arms) != row["arms_sha256"]:
            raise ValueError(f"consensus arm hash mismatch: {source['id']}")
        if row.get("cluster_fallback") is not None:
            fallback += 1
    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "triggered": triggered,
        "trigger_rate": triggered / len(rows),
        "cluster_fallback_rows": fallback,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()