from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluate_pass8_selector import classify
from analyze_pass8_training_bridge import purity_summary, wilson_interval
from build_pass8_deterministic_selector import choose
from run_pass8_selector import extract_selection, packet_digest, walk_forbidden


def test_packet_digest_excludes_only_digest_field() -> None:
    row = {"target_id": "1:2", "candidates": [{"candidate_id": "BASELINE"}]}
    digest = packet_digest(row)
    with_digest = {**row, "packet_sha256": digest}
    assert packet_digest(with_digest) == digest
    assert packet_digest({**with_digest, "target_id": "1:3"}) != digest


def test_recursive_leakage_scan() -> None:
    assert walk_forbidden({"candidates": [{"candidate_id": "C01"}]}) == []
    found = walk_forbidden({"candidates": [{"reward": 1.0}], "gt_action": {"action": "click"}})
    assert "row.candidates[0].reward" in found
    assert "row.gt_action" in found


def test_selection_parser_prefers_tagged_json() -> None:
    parsed = extract_selection(
        'analysis {"candidate_id":"wrong"}\n'
        '<selection>{"candidate_id":"C03","confidence":0.8,"reason":"visible target"}</selection>'
    )
    assert parsed is not None
    assert parsed["candidate_id"] == "C03"


def test_student_relative_outcomes() -> None:
    assert classify(False, True) == "rescue"
    assert classify(True, False) == "regress"
    assert classify(True, True) == "preserve_correct"
    assert classify(False, False) == "unresolved"


def test_deterministic_support_rules() -> None:
    row = {"candidates": [
        {"candidate_id": "BASELINE", "support_count": 1, "source_count": 1},
        {"candidate_id": "C01", "support_count": 3, "neighborhood_support_count": 3, "source_count": 1},
        {"candidate_id": "C02", "support_count": 1, "neighborhood_support_count": 4, "source_count": 2},
    ]}
    assert choose(row, "exact_plurality")[0]["candidate_id"] == "C01"
    assert choose(row, "cross_source_consensus")[0]["candidate_id"] == "C02"


def test_bridge_purity_summary() -> None:
    rows = [
        {"selected_correct": True, "utility_outcome": "rescue"},
        {"selected_correct": False, "utility_outcome": "unresolved"},
        {"selected_correct": False, "utility_outcome": "regress"},
    ]
    summary = purity_summary(rows)
    assert summary["rows"] == 3
    assert summary["selected_correct"] == 1
    assert summary["label_purity"] == 1 / 3
    assert summary["student_relative_net_utility"] == 0
    interval = wilson_interval(1, 3)
    assert interval is not None
    assert interval[0] < 1 / 3 < interval[1]
