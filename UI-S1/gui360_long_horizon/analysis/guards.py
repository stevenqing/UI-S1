"""Hard guards for the GUI-360 history-utilization capstone spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


ALLOWED_RECIPE_DIFF_KEYS = {"dataset", "eval_dataset", "output_dir", "run_name"}


class FormatMismatchError(RuntimeError):
    pass


class RecipeDiffError(RuntimeError):
    pass


class V1UtilizationClaimError(RuntimeError):
    pass


def assert_format_match(arm: str, probe_history_format: str) -> None:
    """Ensure probes use the arm's training history format."""

    expected = {
        "S": {"none"},
        "single_step": {"none"},
        "G": {"gt_history"},
        "gt_history": {"gt_history"},
        "O": {"own_history"},
        "own_history": {"own_history"},
    }.get(arm)
    if expected is None:
        raise FormatMismatchError(f"unknown arm: {arm}")
    if probe_history_format not in expected:
        raise FormatMismatchError(f"format mismatch for arm {arm}: expected {sorted(expected)}, got {probe_history_format!r}")


def assert_no_v1_utilization_claim(verdict_source: str) -> None:
    if verdict_source == "V1":
        raise V1UtilizationClaimError("V1 only certifies OOD repair; history utilization requires V2/V3 evidence")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def changed_keys(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> set[str]:
    keys = set(reference) | set(candidate)
    return {key for key in keys if reference.get(key) != candidate.get(key)}


def assert_recipe_diff_only(reference_yaml: str | Path, arm_yamls: Iterable[str | Path], allowed: set[str] | None = None) -> Dict[str, list[str]]:
    """Assert arm train YAMLs differ only in dataset/output/run_name fields."""

    allowed_keys = allowed or ALLOWED_RECIPE_DIFF_KEYS
    reference = load_yaml(reference_yaml)
    report: Dict[str, list[str]] = {}
    for path in arm_yamls:
        candidate = load_yaml(path)
        diff = changed_keys(reference, candidate)
        bad = sorted(diff - allowed_keys)
        report[str(path)] = sorted(diff)
        if bad:
            raise RecipeDiffError(f"recipe confound in {path}: disallowed diff keys {bad}; allowed={sorted(allowed_keys)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GUI-360 history-utilization hard guards")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recipe = subparsers.add_parser("recipe-diff", help="Assert S/G/O train YAMLs differ only by arm identity fields")
    recipe.add_argument("--reference", required=True, help="Reference S-arm YAML")
    recipe.add_argument("--candidate", action="append", required=True, help="G/O arm YAML; repeat for multiple candidates")

    args = parser.parse_args()
    if args.command == "recipe-diff":
        report = assert_recipe_diff_only(args.reference, args.candidate)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
