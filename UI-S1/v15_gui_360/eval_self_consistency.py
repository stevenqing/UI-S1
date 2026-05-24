#!/usr/bin/env python3
"""Self-consistency and Best-of-N evaluation.

For each step, generate N predictions with temperature > 0.
Analysis modes:
  - Majority voting (action type vote + coordinate clustering)
  - Oracle Best-of-N (upper bound: any of N correct → success)
  - Greedy baseline comparison (from existing eval results)

Key insight: our model has binary grounding (0.5px or 333px).
With N=5 samples, if 3+ find the correct element, majority voting wins.
Effective accuracy: significantly higher than single-shot.

Usage:
    python v15_gui_360/eval_self_consistency.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name balanced_sft_step272 \
        --output_dir v15_gui_360/outputs/preliminary_tests/self_consistency \
        --threads 64 --gt_history --num_samples 5
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot, USER_PROMPT_TEMPLATE,
)
from v13_gui_360.reward import compute_step_reward, _normalize_action_type


def majority_vote_coordinate(coords, threshold=50):
    """Find the coordinate supported by the largest cluster.

    Since grounding is binary (0.5px or 333px), correct predictions
    cluster tightly while wrong ones scatter. Pick the densest cluster.
    """
    if not coords:
        return None
    if len(coords) == 1:
        return coords[0]

    # For each coord, count neighbors within threshold
    best_idx = 0
    best_count = 0
    for i, c1 in enumerate(coords):
        count = sum(
            1 for c2 in coords
            if ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5 < threshold
        )
        if count > best_count:
            best_count = count
            best_idx = i

    # Centroid of the cluster
    center = coords[best_idx]
    cluster = [
        c for c in coords
        if ((center[0] - c[0]) ** 2 + (center[1] - c[1]) ** 2) ** 0.5 < threshold
    ]
    return [
        sum(c[0] for c in cluster) / len(cluster),
        sum(c[1] for c in cluster) / len(cluster),
    ]


def majority_vote_action(parsed_actions):
    """Build a majority-voted action from N parsed actions.

    1. Vote on action type
    2. For clicks: cluster coordinates, pick majority cluster centroid
    3. For type: pick most common text
    4. For swipe: pick most common start/end coordinate pair
    """
    valid = [a for a in parsed_actions if a is not None]
    if not valid:
        return None

    # Vote on action type
    type_counts = Counter(a.get("action", "") for a in valid)
    voted_type = type_counts.most_common(1)[0][0]

    # Filter to actions matching voted type
    typed_actions = [a for a in valid if a.get("action", "") == voted_type]
    if not typed_actions:
        typed_actions = valid

    if voted_type == "click":
        coords = [
            a["coordinate"] for a in typed_actions
            if a.get("coordinate") and a["coordinate"] != [None, None]
        ]
        voted_coord = majority_vote_coordinate(coords) if coords else None
        return {"action": "click", "coordinate": voted_coord}

    elif voted_type == "type":
        texts = [a.get("text", "") for a in typed_actions]
        voted_text = Counter(texts).most_common(1)[0][0] if texts else ""
        coords = [
            a["coordinate"] for a in typed_actions
            if a.get("coordinate") and a["coordinate"] != [None, None]
        ]
        voted_coord = majority_vote_coordinate(coords) if coords else None
        result = {"action": "type", "text": voted_text}
        if voted_coord:
            result["coordinate"] = voted_coord
        return result

    elif voted_type in ("swipe", "drag"):
        starts = [
            a["coordinate"] for a in typed_actions
            if a.get("coordinate") and a["coordinate"] != [None, None]
        ]
        ends = [
            a["endCoordinate"] for a in typed_actions
            if a.get("endCoordinate") and a["endCoordinate"] != [None, None]
        ]
        voted_start = majority_vote_coordinate(starts) if starts else None
        voted_end = majority_vote_coordinate(ends) if ends else None
        return {
            "action": "swipe",
            "coordinate": voted_start,
            "endCoordinate": voted_end,
        }

    else:
        return typed_actions[0]


def evaluate_episode_consistency(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    num_samples: int = 5,
    temperature: float = 0.7,
    gt_history: bool = True,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one episode with N-sample self-consistency."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    history = []
    step_results = []
    correct_majority = 0
    correct_oracle = 0

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        history_text = "\n".join(history) if history else "None"
        b64 = _encode_screenshot(screenshot, image_max_pixels)

        prompt_text = USER_PROMPT_TEMPLATE.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        # Generate N samples
        all_pred_texts = []
        all_parsed = []
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=temperature,
                n=num_samples,
            )
            for choice in response.choices:
                txt = choice.message.content or ""
                all_pred_texts.append(txt)
                parsed = parse_tool_call(txt)
                if parsed is None:
                    m = re.search(r'<action>\s*(\{.*?\})\s*</action>', txt, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group(1))
                        except json.JSONDecodeError:
                            pass
                all_parsed.append(parsed)
        except Exception as e:
            print(f"  [ep {episode_id}] step {i+1} error: {e}")

        # Compute per-sample rewards
        sample_rewards = []
        for parsed in all_parsed:
            if parsed:
                fake = f"<action>{json.dumps(parsed)}</action>"
            else:
                fake = ""
            r, _ = compute_step_reward(fake, gt_action, image_w, image_h)
            sample_rewards.append(r)

        # Oracle Best-of-N: any sample correct?
        oracle_success = any(r >= match_threshold for r in sample_rewards)
        if oracle_success:
            correct_oracle += 1

        # Majority vote
        voted_action = majority_vote_action(all_parsed)
        if voted_action:
            fake_voted = f"<action>{json.dumps(voted_action)}</action>"
        else:
            fake_voted = ""
        vote_reward, vote_info = compute_step_reward(
            fake_voted, gt_action, image_w, image_h
        )
        vote_success = vote_reward >= match_threshold
        if vote_success:
            correct_majority += 1

        step_results.append({
            "step_idx": i,
            "majority_success": vote_success,
            "majority_reward": vote_reward,
            "oracle_success": oracle_success,
            "sample_rewards": sample_rewards,
            "num_correct_samples": sum(1 for r in sample_rewards if r >= match_threshold),
            "voted_type": (voted_action or {}).get("action", ""),
            "gt_type": (vote_info.get("gt_type") or ""),
            "sample_types": [
                _normalize_action_type((p or {}).get("action", ""))
                for p in all_parsed
            ],
        })

        # GT history (teacher-forced)
        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(voted_action, i + 1))

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": num_steps,
        "correct_majority": correct_majority,
        "correct_oracle": correct_oracle,
        "steps": step_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    args = parser.parse_args()

    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes")

    client = OpenAI(base_url=args.api_url, api_key="dummy")
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    total_majority = 0
    total_oracle = 0
    total_steps = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode_consistency, client, args.model_name, ep,
                args.num_samples, args.temperature, args.gt_history,
                args.match_threshold, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="Self-consistency eval")
        for future in as_completed(futures):
            result = future.result()
            eid = result["episode_id"]
            results[eid] = result
            total_majority += result["correct_majority"]
            total_oracle += result["correct_oracle"]
            total_steps += result["num_steps"]

            n = len(results)
            pbar.update(1)
            pbar.set_postfix({
                "MajVote": f"{total_majority/total_steps:.3f}",
                "Oracle": f"{total_oracle/total_steps:.3f}",
            })
        pbar.close()

    # Summary
    n_eps = len(results)
    summary = {
        "num_episodes": n_eps,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "total_steps": total_steps,
        "majority_vote_step_sr": total_majority / total_steps if total_steps > 0 else 0,
        "oracle_best_of_n_step_sr": total_oracle / total_steps if total_steps > 0 else 0,
        "gt_history": args.gt_history,
    }

    # Per-type breakdown
    type_stats = defaultdict(lambda: {"majority": 0, "oracle": 0, "total": 0})
    agreement_stats = {"full_agree": 0, "partial": 0, "no_agree": 0, "total": 0}

    for eid, ep in results.items():
        for s in ep["steps"]:
            gt_type = s.get("gt_type", "")
            type_stats[gt_type]["total"] += 1
            if s["majority_success"]:
                type_stats[gt_type]["majority"] += 1
            if s["oracle_success"]:
                type_stats[gt_type]["oracle"] += 1

            # Agreement analysis
            n_correct = s["num_correct_samples"]
            agreement_stats["total"] += 1
            if n_correct == args.num_samples:
                agreement_stats["full_agree"] += 1
            elif n_correct > 0:
                agreement_stats["partial"] += 1
            else:
                agreement_stats["no_agree"] += 1

    summary["type_breakdown"] = dict(type_stats)
    summary["agreement"] = agreement_stats

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Self-Consistency Results (N={args.num_samples}, T={args.temperature})")
    print(f"{'='*60}")
    print(f"  Majority Vote Step SR:  {summary['majority_vote_step_sr']*100:.1f}%")
    print(f"  Oracle Best-of-N SR:    {summary['oracle_best_of_n_step_sr']*100:.1f}%")
    print(f"  (Baseline greedy ~58.8%)")
    print()

    # Type breakdown
    for gt_type in ["click", "type", "swipe"]:
        s = type_stats.get(gt_type, {"majority": 0, "oracle": 0, "total": 0})
        if s["total"] > 0:
            print(f"  {gt_type:<8}: Majority={s['majority']}/{s['total']}={s['majority']/s['total']*100:.1f}%"
                  f"  Oracle={s['oracle']}/{s['total']}={s['oracle']/s['total']*100:.1f}%")

    print()
    t = agreement_stats["total"]
    print(f"  Agreement: all_correct={agreement_stats['full_agree']}/{t}"
          f"  partial={agreement_stats['partial']}/{t}"
          f"  all_wrong={agreement_stats['no_agree']}/{t}")

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
