#!/usr/bin/env python3
"""Multi-agent complementarity on GUI-360 critical steps.

Inference-only. Uses cached SFT/base rows when available and can sample other
OpenAI-compatible multimodal endpoints on the same critical-step set.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.minimal_validation import build_messages, parse_prediction, read_jsonl  # noqa: E402
from scripts.rl_feasibility_sampling import action_key, sanitize_jsonable  # noqa: E402
from v13_gui_360.reward import compute_step_reward  # noqa: E402


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_jsonable(dict(row)), ensure_ascii=False) + "\n")


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:+.2f}pp"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def target_id(row: Mapping[str, Any]) -> str:
    return str(row.get("target_id") or f"{row['episode_id']}:{row['step_idx']}")


def load_target_ids(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(item) for item in data.get("target_ids") or data.get("critical_ids") or []}
    return {str(item) for item in data}


def choose_targets(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.source))
    if args.strong_reflex:
        rows = sorted(rows, key=lambda row: (float(row.get("greedy_decode_share") or 0.0), target_id(row)))
        rows = rows[len(rows) // 2:]
    rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("p_i_initial") if row.get("p_i_initial") is not None else 1.0),
            -float(row.get("greedy_decode_share") or 0.0),
            str(row.get("episode_id")),
            int(row.get("step_idx") or 0),
        ),
    )
    if args.max_steps > 0:
        rows = rows[: args.max_steps]
    payload = {
        "source": args.source,
        "strong_reflex": args.strong_reflex,
        "max_steps": args.max_steps,
        "target_ids": [target_id(row) for row in rows],
        "rows": [{
            "target_id": target_id(row),
            "episode_id": str(row.get("episode_id")),
            "step_idx": int(row.get("step_idx") or 0),
            "p_i_initial": row.get("p_i_initial"),
            "greedy_decode_share": row.get("greedy_decode_share"),
            "greedy_correct": bool(row.get("greedy_correct")),
            "critical_source": row.get("critical_source") or [],
        } for row in rows],
    }
    write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "targets": len(payload["target_ids"])}, indent=2), flush=True)


def image_to_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")


def call_chat(api_url: str, model: str, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float, top_p: float, timeout: int, chat_template_kwargs: dict[str, Any] | None) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    response = requests.post(api_url.rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer EMPTY"}, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    return f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content


def score_text(text: str, gt_action: Mapping[str, Any], image_w: int, image_h: int, coord_bucket: int, match_threshold: float) -> dict[str, Any]:
    try:
        pred_action = parse_prediction(text)
    except Exception:
        pred_action = None
    fake_text = f"<action>{json.dumps(pred_action, ensure_ascii=False)}</action>" if pred_action else text
    reward, info = compute_step_reward(fake_text, dict(gt_action), image_w=image_w, image_h=image_h)
    pred = info.get("pred_action")
    return {
        "raw_output": text[:1000],
        "reward": float(reward),
        "correct": bool(reward >= match_threshold),
        "pred_type": info.get("pred_type"),
        "gt_type": info.get("gt_type"),
        "pred_action": pred,
        "action_key": action_key(pred, coord_bucket),
        "parse_ok": pred is not None,
    }


def sample_one(episode: Mapping[str, Any], step_idx: int, args: argparse.Namespace, api_url: str) -> dict[str, Any]:
    step = episode["steps"][step_idx]
    messages, image_w, image_h = build_messages(str(episode.get("goal") or ""), list(step.get("history") or []), step["screenshot"], args.image_max_pixels)
    # build_messages uses current history only when provided by the step; exported episodes do not store history.
    # Rebuild GT history here for the original GUI-360 teacher-forced prompt.
    from v13_gui_360.eval_gui360_template import _format_action_for_history
    history = [_format_action_for_history(prev.get("action", {}) or {}, index + 1) for index, prev in enumerate(episode.get("steps", [])[:step_idx])]
    messages, image_w, image_h = build_messages(str(episode.get("goal") or ""), history, step["screenshot"], args.image_max_pixels)
    texts: list[str] = []
    for temp in [0.0] + [args.sample_temperature] * max(0, args.n_candidates - 1):
        last_exc: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                texts.append(call_chat(
                    api_url,
                    args.model_name,
                    messages,
                    max_tokens=args.max_tokens,
                    temperature=temp,
                    top_p=1.0 if temp == 0.0 else args.top_p,
                    timeout=args.request_timeout,
                    chat_template_kwargs=args.chat_template_kwargs,
                ))
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(min(2.0 * (attempt + 1), 10.0))
        else:
            texts.append(f"ERROR: {last_exc}")
    samples = [score_text(text, step["action"], image_w, image_h, args.coord_bucket, args.match_threshold) for text in texts]
    keys = [sample["action_key"] for sample in samples]
    key_counts = Counter(keys)
    correct_by_key: Counter[str] = Counter()
    for sample in samples:
        if sample["correct"]:
            correct_by_key[sample["action_key"]] += 1
    best_correct_share = (max(correct_by_key.values()) / max(1, len(samples))) if correct_by_key else 0.0
    greedy_key = keys[0] if keys else "__missing__"
    return {
        "target_id": f"{episode['episode_id']}:{step_idx}",
        "episode_id": str(episode["episode_id"]),
        "step_idx": step_idx,
        "num_steps": len(episode.get("steps", [])),
        "gt_action": step["action"],
        "image_w": image_w,
        "image_h": image_h,
        "agent": args.agent,
        "tier_hint": args.tier_hint,
        "model_name": args.model_name,
        "sample_count": len(samples),
        "parse_count": sum(1 for sample in samples if sample["parse_ok"]),
        "parse_rate": sum(1 for sample in samples if sample["parse_ok"]) / max(1, len(samples)),
        "greedy_correct": bool(samples[0]["correct"]) if samples else False,
        "greedy_action_key": greedy_key,
        "greedy_decode_share": key_counts[greedy_key] / max(1, len(samples)),
        "any_correct": any(sample["correct"] for sample in samples),
        "best_correct_share": best_correct_share,
        "confident_correct": best_correct_share >= args.conf_threshold,
        "confident_wrong": (not bool(samples[0]["correct"])) and (key_counts[greedy_key] / max(1, len(samples)) >= args.conf_threshold),
        "samples": samples,
    }


def sample_agent(args: argparse.Namespace) -> None:
    target_ids = load_target_ids(Path(args.targets))
    episodes = {str(row["episode_id"]): row for row in read_jsonl(Path(args.episode_data))}
    jobs: list[tuple[str, int]] = []
    for target in sorted(target_ids, key=lambda value: (int(value.split(":")[0]) if value.split(":")[0].isdigit() else value, int(value.split(":")[-1]))):
        episode_id, step_text = target.split(":")
        if episode_id in episodes:
            jobs.append((episode_id, int(step_text)))
    done = {row.get("target_id") for row in read_jsonl(Path(args.output)) if row.get("target_id")}
    pending = [job for job in jobs if f"{job[0]}:{job[1]}" not in done]
    print(json.dumps({"agent": args.agent, "jobs": len(jobs), "done": len(done), "pending": len(pending), "api_urls": args.api_urls}, indent=2), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    url_count = len(args.api_urls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = []
        for index, (episode_id, step_idx) in enumerate(pending):
            futures.append(pool.submit(sample_one, episodes[episode_id], step_idx, args, args.api_urls[index % url_count]))
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            append_jsonl(Path(args.output), row)
            if index % 25 == 0:
                print(f"{args.agent} completed {index}/{len(pending)}", flush=True)
    print(json.dumps({"agent": args.agent, "output": args.output, "rows": len(read_jsonl(Path(args.output)))}, indent=2), flush=True)


def normalize_cached_row(row: Mapping[str, Any], *, agent: str, tier_hint: str, conf_threshold: float, coord_bucket: int) -> dict[str, Any]:
    samples = []
    if row.get("greedy_pred_action") is not None:
        samples.append({
            "correct": bool(row.get("greedy_correct")),
            "parse_ok": row.get("greedy_pred_action") is not None,
            "action_key": action_key(row.get("greedy_pred_action"), coord_bucket),
            "reward": float(row.get("greedy_reward") or 0.0),
            "pred_action": row.get("greedy_pred_action"),
            "raw_output": "__cached_greedy__",
        })
    for sample in row.get("samples") or []:
        samples.append({
            "correct": bool(sample.get("correct")),
            "parse_ok": sample.get("pred_action") is not None,
            "action_key": sample.get("action_key") or action_key(sample.get("pred_action"), coord_bucket),
            "reward": float(sample.get("reward") or 0.0),
            "pred_action": sample.get("pred_action"),
            "raw_output": sample.get("raw_output", "")[:1000],
        })
    if not samples and row.get("sample_count"):
        for sample in row.get("samples") or []:
            samples.append(sample)
    key_counts = Counter(sample.get("action_key", "__missing__") for sample in samples)
    correct_counts = Counter(sample.get("action_key", "__missing__") for sample in samples if sample.get("correct"))
    greedy_key = samples[0].get("action_key", "__missing__") if samples else "__missing__"
    best_correct_share = max(correct_counts.values()) / max(1, len(samples)) if correct_counts else 0.0
    return {
        "target_id": target_id(row),
        "episode_id": str(row.get("episode_id")),
        "step_idx": int(row.get("step_idx") or 0),
        "num_steps": int(row.get("num_steps") or 0),
        "gt_action": row.get("gt_action"),
        "agent": agent,
        "tier_hint": tier_hint,
        "sample_count": len(samples),
        "parse_count": sum(1 for sample in samples if sample.get("parse_ok", True)),
        "parse_rate": sum(1 for sample in samples if sample.get("parse_ok", True)) / max(1, len(samples)),
        "greedy_correct": bool(row.get("greedy_correct")) if "greedy_correct" in row else bool(samples[0].get("correct")) if samples else False,
        "greedy_action_key": greedy_key,
        "greedy_decode_share": key_counts[greedy_key] / max(1, len(samples)),
        "any_correct": any(sample.get("correct") for sample in samples),
        "best_correct_share": best_correct_share,
        "confident_correct": best_correct_share >= conf_threshold,
        "confident_wrong": (not (bool(row.get("greedy_correct")) if "greedy_correct" in row else bool(samples[0].get("correct")) if samples else False)) and (key_counts[greedy_key] / max(1, len(samples)) >= conf_threshold),
        "samples": samples,
    }


def load_agent_rows(spec: str, target_ids: set[str], conf_threshold: float, coord_bucket: int) -> tuple[str, str, list[dict[str, Any]]]:
    # spec: agent:tier:path
    agent, tier, path_text = spec.split(":", 2)
    rows = [normalize_cached_row(row, agent=agent, tier_hint=tier, conf_threshold=conf_threshold, coord_bucket=coord_bucket) for row in read_jsonl(Path(path_text))]
    rows = [row for row in rows if row["target_id"] in target_ids]
    return agent, tier, rows


def agent_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "steps": n,
        "parse_rate": sum(row["parse_rate"] for row in rows) / max(1, n),
        "greedy_accuracy": sum(1 for row in rows if row["greedy_correct"]) / max(1, n),
        "any_correct_coverage": sum(1 for row in rows if row["any_correct"]) / max(1, n),
        "confident_correct_coverage": sum(1 for row in rows if row["confident_correct"]) / max(1, n),
        "confident_wrong_fraction": sum(1 for row in rows if row["confident_wrong"]) / max(1, n),
        "noise_flag": n == 0 or (sum(row["parse_rate"] for row in rows) / max(1, n) < 0.5) or (sum(1 for row in rows if row["greedy_correct"]) / max(1, n) < 0.02),
    }


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def tier_metrics(agent_names: list[str], by_agent: dict[str, dict[str, dict[str, Any]]], target_ids: set[str], high_temp: dict[str, Any] | None) -> dict[str, Any]:
    if not agent_names:
        return {"available": False, "agents": []}
    failures = {name: {tid for tid, row in by_agent[name].items() if not row["greedy_correct"]} for name in agent_names}
    pairwise = []
    for i, left in enumerate(agent_names):
        for right in agent_names[i + 1:]:
            pairwise.append({"pair": [left, right], "failure_jaccard": jaccard(failures[left], failures[right])})
    all_fail = sum(1 for tid in target_ids if all(tid in failures[name] for name in agent_names))
    some_fail = sum(1 for tid in target_ids if any(tid in failures[name] for name in agent_names))
    union_conf = {tid for tid in target_ids if any(by_agent[name].get(tid, {}).get("confident_correct") for name in agent_names)}
    union_any = {tid for tid in target_ids if any(by_agent[name].get(tid, {}).get("any_correct") for name in agent_names)}
    single_conf = {name: sum(1 for tid in target_ids if by_agent[name].get(tid, {}).get("confident_correct")) / max(1, len(target_ids)) for name in agent_names}
    best_single_agent = max(single_conf, key=single_conf.get)
    meaningful = lucky = 0
    for tid in target_ids:
        for a in agent_names:
            row_a = by_agent[a].get(tid, {})
            if not row_a.get("confident_wrong"):
                continue
            for b in agent_names:
                if a == b:
                    continue
                row_b = by_agent[b].get(tid, {})
                if row_b.get("confident_correct"):
                    meaningful += 1
                elif row_b.get("any_correct"):
                    lucky += 1
    union_cov = len(union_conf) / max(1, len(target_ids))
    high_temp_cov = high_temp.get("confident_correct_coverage") if high_temp else None
    return {
        "available": True,
        "agents": agent_names,
        "pairwise_failure_jaccard": pairwise,
        "mean_pairwise_failure_jaccard": sum(item["failure_jaccard"] for item in pairwise) / max(1, len(pairwise)),
        "all_fail_fraction": all_fail / max(1, len(target_ids)),
        "some_fail_fraction": some_fail / max(1, len(target_ids)),
        "single_confident_correct": single_conf,
        "best_single_agent": best_single_agent,
        "best_single_confident_correct": single_conf[best_single_agent],
        "union_confident_correct": union_cov,
        "union_any_correct": len(union_any) / max(1, len(target_ids)),
        "lift_vs_best_single": union_cov - single_conf[best_single_agent],
        "high_temp_confident_correct": high_temp_cov,
        "lift_vs_sft_high_temp": None if high_temp_cov is None else union_cov - high_temp_cov,
        "meaningful_complementarity_events": meaningful,
        "lucky_noise_events": lucky,
    }


def report(args: argparse.Namespace) -> None:
    target_ids = load_target_ids(Path(args.targets))
    agent_rows: dict[str, list[dict[str, Any]]] = {}
    agent_tiers: dict[str, str] = {}
    for spec in args.agent_rows:
        agent, tier, rows = load_agent_rows(spec, target_ids, args.conf_threshold, args.coord_bucket)
        agent_rows[agent] = rows
        agent_tiers[agent] = tier
    by_agent = {agent: {row["target_id"]: row for row in rows} for agent, rows in agent_rows.items()}
    metrics_by_agent = {agent: agent_metrics(rows) for agent, rows in agent_rows.items()}

    high_temp = None
    if args.sft_high_temp:
        _, _, high_rows = load_agent_rows(f"sft_high_temp:single_policy:{args.sft_high_temp}", target_ids, args.conf_threshold, args.coord_bucket)
        high_temp = agent_metrics(high_rows)

    tiers = {
        "tier1_near": [agent for agent, tier in agent_tiers.items() if tier == "tier1_near"],
        "tier2_half": [agent for agent, tier in agent_tiers.items() if tier == "tier2_half"],
        "tier3_cross": [agent for agent, tier in agent_tiers.items() if tier == "tier3_cross"],
    }
    metrics_by_tier = {tier: tier_metrics(agents, by_agent, target_ids, high_temp) for tier, agents in tiers.items()}

    per_step_rows = []
    for tid in sorted(target_ids, key=lambda value: (int(value.split(":")[0]) if value.split(":")[0].isdigit() else value, int(value.split(":")[-1]))):
        step_agents = {agent: by_agent.get(agent, {}).get(tid) for agent in agent_rows}
        per_step_rows.append({
            "target_id": tid,
            "agents": step_agents,
            "tier1_union_confident_correct": any((step_agents.get(agent) or {}).get("confident_correct") for agent in tiers["tier1_near"]),
            "tier1_all_fail": all(not (step_agents.get(agent) or {}).get("greedy_correct") for agent in tiers["tier1_near"]) if tiers["tier1_near"] else None,
        })
    output_dir = Path(args.output_dir)
    write_jsonl = output_dir / "per_step.jsonl"
    write_jsonl.write_text("", encoding="utf-8")
    for row in per_step_rows:
        append_jsonl(write_jsonl, row)

    gate = "GUI TENSION BLOCKS IT (honest negative)"
    reason = "Only near-tier Qwen-lineage agents are available locally; no half/cross lineage checkpoints were found."
    for tier, metric in metrics_by_tier.items():
        if metric.get("available") and metric.get("lift_vs_best_single", 0.0) >= args.lift_threshold and (metric.get("lift_vs_sft_high_temp") is None or metric.get("lift_vs_sft_high_temp") >= 0.0):
            gate = "MEANINGFUL COMPLEMENTARITY AT SOME TIER (multi-agent has value there)"
            reason = f"{tier} union confident-correct coverage beats best single by at least {pct(args.lift_threshold)}."
            break
    if gate.startswith("GUI") and metrics_by_tier["tier1_near"].get("available"):
        near = metrics_by_tier["tier1_near"]
        if near.get("lift_vs_best_single", 0.0) < args.lift_threshold:
            reason = "Near-tier Qwen-lineage agents are available but show limited union lift; half/cross tiers are unavailable locally for this run."

    summary = {
        "gate": gate,
        "reason": reason,
        "targets": len(target_ids),
        "conf_threshold": args.conf_threshold,
        "agent_metrics": metrics_by_agent,
        "tier_metrics": metrics_by_tier,
        "sft_high_temp": high_temp,
        "unavailable_tiers": {
            "tier2_half": "No InternVL checkpoint found locally.",
            "tier3_cross": "No LLaVA/Llama-Vision/Molmo/Pixtral checkpoint found locally.",
        },
    }
    write_json(output_dir / "summary.json", summary)

    lines = [
        "# Multi-Agent Complementarity On Critical Steps",
        "",
        "Frozen matcher; inference only; no training. Confidence is action decode-share over greedy plus sampled candidates; confident-correct requires a correct action key share above the configured threshold.",
        "",
        "## Run Scope",
        "",
        table(["field", "value"], [
            ["targets", len(target_ids)],
            ["confidence threshold", args.conf_threshold],
            ["available local tiers", ", ".join(t for t, agents in tiers.items() if agents) or "none"],
            ["unavailable HALF", "InternVL not found locally"],
            ["unavailable CROSS", "LLaVA/Llama-Vision/Molmo/Pixtral not found locally"],
        ]),
        "",
        "## Metric 0 - Per-Agent Fair Ability",
        "",
        table(["agent", "tier", "steps", "parse", "greedy acc", "any correct", "conf-correct", "conf-wrong", "noise flag"], [
            [agent, agent_tiers[agent], m["steps"], pct(m["parse_rate"]), pct(m["greedy_accuracy"]), pct(m["any_correct_coverage"]), pct(m["confident_correct_coverage"]), pct(m["confident_wrong_fraction"]), m["noise_flag"]]
            for agent, m in metrics_by_agent.items()
        ]),
        "",
    ]
    if high_temp:
        lines.extend(["Single-SFT high-temperature reference:", "", table(["metric", "value"], [["conf-correct", pct(high_temp["confident_correct_coverage"])], ["any correct", pct(high_temp["any_correct_coverage"])], ["parse", pct(high_temp["parse_rate"])]]) , ""])
    lines.extend(["## Per-Tier Metrics", ""])
    for tier, metric in metrics_by_tier.items():
        lines.extend([f"### {tier}", ""])
        if not metric.get("available"):
            lines.extend(["Unavailable in this run.", ""])
            continue
        lines.extend([
            table(["metric", "value"], [
                ["agents", ", ".join(metric["agents"])],
                ["mean failure Jaccard", pct(metric["mean_pairwise_failure_jaccard"])],
                ["all fail fraction", pct(metric["all_fail_fraction"])],
                ["some fail fraction", pct(metric["some_fail_fraction"])],
                ["best single", f"{metric['best_single_agent']} / {pct(metric['best_single_confident_correct'])}"],
                ["union confident-correct", pct(metric["union_confident_correct"])],
                ["union any-correct", pct(metric["union_any_correct"])],
                ["lift vs best single", pp(metric["lift_vs_best_single"])],
                ["lift vs SFT high-temp", pp(metric.get("lift_vs_sft_high_temp"))],
                ["meaningful events", metric["meaningful_complementarity_events"]],
                ["lucky/noise events", metric["lucky_noise_events"]],
            ]),
            "",
        ])
    lines.extend(["## Gate", "", gate, "", reason, "", "STOP for review.", ""])
    (output_dir / "complementarity.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "gate": gate}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-targets")
    p.add_argument("--source", default="outputs/rl_feasibility/per_step.jsonl")
    p.add_argument("--output", default="outputs/multiagent_complementarity/target_ids.json")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--strong-reflex", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=choose_targets)

    p = sub.add_parser("sample-agent")
    p.add_argument("--agent", required=True)
    p.add_argument("--tier-hint", default="tier1_near")
    p.add_argument("--episode-data", default="outputs/validation_2k/data/test_episodes.jsonl")
    p.add_argument("--targets", default="outputs/multiagent_complementarity/target_ids.json")
    p.add_argument("--output", required=True)
    p.add_argument("--api-urls", nargs="+", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--sample-temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--image-max-pixels", type=int, default=602112)
    p.add_argument("--match-threshold", type=float, default=0.5)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--coord-bucket", type=int, default=25)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--request-timeout", type=int, default=180)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    p.set_defaults(func=lambda args: (setattr(args, "chat_template_kwargs", {"enable_thinking": bool(args.enable_thinking)}), sample_agent(args))[1])

    p = sub.add_parser("report")
    p.add_argument("--targets", default="outputs/multiagent_complementarity/target_ids.json")
    p.add_argument("--output-dir", default="outputs/multiagent_complementarity")
    p.add_argument("--agent-rows", nargs="+", required=True, help="agent:tier:path")
    p.add_argument("--sft-high-temp", default="outputs/temp_restores_signal/sft_T1p5_critical.jsonl")
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--coord-bucket", type=int, default=25)
    p.add_argument("--lift-threshold", type=float, default=0.02)
    p.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()