#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bench.common import summarize_assets
from src.config import REPO_ROOT, load_config, resolve_path, write_json
from src.metrics.audit import truncation_summary
from src.reports.gate_a import write_gate_a_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase A baseline reproduction/preflight for CHORUS N0/N1")
    parser.add_argument("--config", action="append", required=True, help="Path to Phase A YAML config. Repeat for multiple benchmarks.")
    parser.add_argument("--preflight-only", action="store_true", help="Only check prerequisites and write Gate A report")
    args = parser.parse_args()

    preflights = []
    for config_path in args.config:
        cfg = load_config(config_path)
        run_id = f"{cfg.get('run', {}).get('name', 'phase_a')}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = resolve_path(Path(cfg.get("run", {}).get("output_root", "chorus-n0n1/runs")) / run_id)
        assert run_dir is not None
        run_dir.mkdir(parents=True, exist_ok=True)

        preflight = build_preflight(cfg)
        preflight["run_id"] = run_id
        preflight["run_dir"] = str(run_dir)
        write_json(run_dir / "preflight.json", preflight)
        preflights.append(preflight)
        print(f"Wrote preflight: {run_dir / 'preflight.json'}")

    report_payload = preflights[0] if len(preflights) == 1 else {"benchmarks": preflights}
    report_path = write_gate_a_report(report_payload)
    print(f"Wrote Gate A report: {report_path}")

    has_blockers = any(p.get("blocking_issues") for p in preflights)
    if args.preflight_only:
        return 2 if has_blockers else 0

    if has_blockers:
        print("Phase A baseline reproduction is blocked. See Gate A report.")
        return 2

    print("Baseline execution is not enabled in this scaffold until the official HiconAgent eval repo is configured.")
    return 2


def build_preflight(cfg: Dict[str, Any]) -> Dict[str, Any]:
    benchmark = cfg.get("benchmark", {})
    model = cfg.get("model", {})
    official = cfg.get("official", {})
    phase = cfg.get("phase_a", {})

    checks: List[Dict[str, Any]] = []
    blocking: List[str] = []

    jsonl_path = resolve_path(benchmark.get("jsonl"))
    records = []
    if jsonl_path and jsonl_path.exists():
        try:
            if benchmark.get("name") == "android_control":
                from src.bench.android_control import load_steps
            elif benchmark.get("name") == "gui_odyssey":
                from src.bench.gui_odyssey import load_steps
            else:
                load_steps = None
            if load_steps is None:
                raise ValueError(f"unsupported benchmark {benchmark.get('name')}")
            records = load_steps(jsonl_path, benchmark.get("split", "test"))
            checks.append({"name": "benchmark_jsonl", "ok": True, "detail": f"{jsonl_path} ({len(records)} steps)"})
        except Exception as exc:
            checks.append({"name": "benchmark_jsonl", "ok": False, "detail": f"failed to load {jsonl_path}: {exc!r}"})
            blocking.append(f"Benchmark JSONL failed to load: `{jsonl_path}`.")
    else:
        checks.append({"name": "benchmark_jsonl", "ok": False, "detail": str(jsonl_path)})
        blocking.append(f"Benchmark JSONL is missing: `{jsonl_path}`.")

    if records:
        asset_summary = summarize_assets(records)
        ok = bool(asset_summary.get("screenshots_available"))
        checks.append({"name": "screenshots", "ok": ok, "detail": json.dumps(asset_summary, ensure_ascii=False)})
        if not ok:
            examples = asset_summary.get("missing_screenshot_examples", [])
            blocking.append(f"Benchmark screenshots are missing. Examples: `{examples}`.")

    ckpt = resolve_path(model.get("checkpoint_path"))
    model_id = model.get("id") or model.get("served_name") or "model"
    if ckpt and ckpt.exists():
        checks.append({"name": "model_checkpoint", "ok": True, "detail": str(ckpt)})
    else:
        checks.append({"name": "model_checkpoint", "ok": False, "detail": str(ckpt)})
        blocking.append(f"{model_id} checkpoint is not configured locally (`model.checkpoint_path`).")

    official_path = resolve_path(official.get("local_path"))
    eval_entry = official.get("eval_entry")
    if official_path and official_path.exists():
        entry_path = official_path / eval_entry if eval_entry else official_path
        ok = entry_path.exists()
        checks.append({"name": "official_repo", "ok": ok, "detail": str(entry_path)})
        if not ok:
            blocking.append(f"Official eval entry is missing: `{entry_path}`.")
    else:
        checks.append({"name": "official_repo", "ok": False, "detail": str(official_path)})
        blocking.append("Official repository is not configured locally (`official.local_path`).")

    api_url = str(model.get("api_url", ""))
    checks.append({"name": "api_url_configured", "ok": bool(api_url), "detail": api_url})
    api_port_open = _port_open(api_url)
    checks.append({"name": "api_port_open", "ok": api_port_open, "detail": api_url})
    if not api_port_open:
        blocking.append(f"OpenAI-compatible inference endpoint is not reachable: `{api_url}`.")

    require_logprobs = bool(phase.get("require_logprobs", True))
    logprobs_ok = bool(model.get("logprobs"))
    checks.append({"name": "logprobs_requested", "ok": (not require_logprobs) or logprobs_ok, "detail": str(model.get("logprobs"))})
    if require_logprobs and not logprobs_ok:
        blocking.append("Action-token logprobs are required for N1 baseline detector but `model.logprobs` is false.")

    return {
        "config_path": cfg.get("_config_path"),
        "benchmark": benchmark,
        "model": {k: v for k, v in model.items() if k != "api_key"},
        "official": official,
        "checks": checks,
        "blocking_issues": blocking,
        "truncation_summary": truncation_summary([]),
        "qualitative_examples": [],
    }


def _port_open(api_url: str) -> bool:
    if not api_url.startswith("http"):
        return False
    try:
        host_port = api_url.split("//", 1)[1].split("/", 1)[0]
        host, port_text = host_port.rsplit(":", 1)
        with socket.create_connection((host, int(port_text)), timeout=1.0):
            return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
