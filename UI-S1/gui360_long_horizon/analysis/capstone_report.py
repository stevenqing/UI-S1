"""Aggregate capstone probe summaries and write verdict JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from gui360_long_horizon.analysis.stats import capstone_decision


DEFAULT_VERDICT_PATH = "gui360_long_horizon/reports/verdict.json"
DEFAULT_RESULTS_PATH = "gui360_long_horizon/reports/capstone_results.json"


def load_results(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"capstone results must be a JSON object: {path}")
    return data


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON summary must be an object: {path}")
    return data


def load_probe_summaries(summary_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    root = Path(summary_dir)
    if not root.exists():
        raise FileNotFoundError(f"summary_dir does not exist: {root}")
    summaries: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        summaries[path.stem] = _load_json(path)
    if not summaries:
        raise FileNotFoundError(f"no *.json summaries under {root}")
    return summaries


def _pick_probe(summaries: Dict[str, Dict[str, Any]], probe: str, preferred_arms: Iterable[str]) -> Optional[Dict[str, Any]]:
    for arm in preferred_arms:
        key = f"{arm}_{probe}"
        if key in summaries:
            return summaries[key]
    return None


def aggregate_probe_summaries(
    summary_dir: str | Path,
    *,
    preferred_arms: Iterable[str] = ("O", "G"),
    required_v1_arms: Iterable[str] = ("G", "O"),
    v3_summary: str | Path | None = None,
) -> Dict[str, Any]:
    """Aggregate per-arm V1/V2/V4 summaries into the §5 decision input."""

    summaries = load_probe_summaries(summary_dir)
    results: Dict[str, Any] = {"summaries": summaries}

    required = [arm for arm in required_v1_arms if f"{arm}_v1" in summaries]
    if required:
        results["v1"] = {"repaired": all(bool(summaries[f"{arm}_v1"].get("repaired")) for arm in required), "required_arms": required}
        for arm in required:
            results["v1"][arm] = summaries[f"{arm}_v1"]
    elif "G_v1" in summaries or "O_v1" in summaries:
        key = "O_v1" if "O_v1" in summaries else "G_v1"
        results["v1"] = dict(summaries[key])

    if "G_v1" in summaries and "O_v1" in summaries:
        results["og_contrast"] = {
            "value": float(summaries["O_v1"].get("matched_minus_none", 0.0)) - float(summaries["G_v1"].get("matched_minus_none", 0.0)),
            "definition": "O_v1.matched_minus_none - G_v1.matched_minus_none",
        }

    for probe, result_key in (("v2", "v2"), ("v4", "v4")):
        picked = _pick_probe(summaries, probe, preferred_arms)
        if picked is not None:
            results[result_key] = picked

    if v3_summary is not None:
        results["v3"] = _load_json(v3_summary)
    elif "v3" in summaries:
        results["v3"] = summaries["v3"]
    elif "G_v3" in summaries or "O_v3" in summaries:
        picked = _pick_probe(summaries, "v3", preferred_arms)
        if picked is not None:
            results["v3"] = picked
    return results


def write_results(results: Dict[str, Any], output_path: str | Path = DEFAULT_RESULTS_PATH) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_verdict(results: Dict[str, Any], output_path: str | Path = DEFAULT_VERDICT_PATH) -> Path:
    verdict = capstone_decision(results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"verdict": asdict(verdict), "inputs": results}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate GUI-360 History-Utilization A/B probes and write verdict.json")
    parser.add_argument("--results", help="JSON object containing v1/v2/v3/v4 summaries")
    parser.add_argument("--summary-dir", help="Directory containing per-arm summaries such as G_v1.json and O_v4.json")
    parser.add_argument("--v3-summary", help="Optional V3 long-dependency summary JSON")
    parser.add_argument("--results-out", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--out", default=DEFAULT_VERDICT_PATH)
    args = parser.parse_args()
    if args.summary_dir:
        results = aggregate_probe_summaries(args.summary_dir, v3_summary=args.v3_summary)
        results_path = write_results(results, args.results_out)
    elif args.results:
        results = load_results(args.results)
        results_path = Path(args.results)
    else:
        raise SystemExit("either --results or --summary-dir is required")
    verdict_path = write_verdict(results, args.out)
    print(json.dumps({"results": str(results_path), "verdict": str(verdict_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()