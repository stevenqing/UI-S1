#!/usr/bin/env python3
"""Prepare Action-Space GRPO data by perturbing action parameters.

Core insight: Standard GRPO fails for GUI agents because temperature sampling
in TOKEN SPACE doesn't produce diversity in ACTION SPACE. All candidates click
the same location / type the same text, producing zero reward variance.

Solution: Perturb actions in the ACTION PARAMETER SPACE directly:
  - Click: Gaussian perturbation on (x, y) coordinates → distance-based reward
  - Type:  Character-level text perturbation → edit-distance-based reward
  - Swipe: skipped (insufficient GT structure)

This creates candidates that are genuinely different in action space,
guaranteeing reward variance for GRPO to learn from.

Usage:
  python v19_step_aware/prepare_as_grpo_data.py \
    --input v19_step_aware/data/grpo_candidates_n5.jsonl \
    --output v19_step_aware/data/as_grpo_candidates_n8.jsonl \
    --n_candidates 8 --coord_sigma 40 --reward_sigma 0.05
"""

import argparse
import json
import logging
import re
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("as_grpo_data")


# ═══════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════════════════════════════════

def _parse_stored(obj):
    """Parse a stored action that may be dict or string."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        import ast
        try:
            return ast.literal_eval(obj)
        except Exception:
            pass
        try:
            return json.loads(obj.replace("'", '"'))
        except Exception:
            pass
    return {}


# ═══════════════════════════════════════════════════════════════════════
# Click: coordinate perturbation
# ═══════════════════════════════════════════════════════════════════════

def _replace_coordinate(response_text, new_coord):
    """Replace coordinate values in the <tool_call> block."""
    tc_start = response_text.find("<tool_call>")
    if tc_start < 0:
        return response_text
    before = response_text[:tc_start]
    after = response_text[tc_start:]
    pattern = r'("coordinate"\s*:\s*\[)\s*[\d.]+\s*,\s*[\d.]+\s*(\])'
    replacement = f'\\g<1>{int(new_coord[0])}, {int(new_coord[1])}\\g<2>'
    return before + re.sub(pattern, replacement, after, count=1)


def _gaussian_coord_reward(pred_coord, gt_coord, sigma, img_w, img_h):
    """Gaussian distance reward for click coordinates."""
    try:
        dx = (float(pred_coord[0]) - float(gt_coord[0])) / img_w
        dy = (float(pred_coord[1]) - float(gt_coord[1])) / img_h
        dist = (dx ** 2 + dy ** 2) ** 0.5
    except (TypeError, IndexError, ValueError):
        return 0.0
    return float(np.exp(-(dist / sigma) ** 2))


def _perturb_click(greedy_cand, gt_coord, n_candidates, coord_sigma,
                   reward_sigma, img_w, img_h, rng):
    """Generate N click candidates via coordinate perturbation."""
    pred_action = _parse_stored(greedy_cand.get("pred_action", {}))
    pred_coord = pred_action.get("coordinate")
    if pred_coord is None:
        return None
    pred_x, pred_y = float(pred_coord[0]), float(pred_coord[1])

    candidates = []
    rewards = []

    # Candidate 0: original greedy
    r0 = _gaussian_coord_reward(pred_coord, gt_coord, reward_sigma, img_w, img_h)
    candidates.append({
        "response": greedy_cand["response"],
        "reward": 0.3 + 0.7 * r0,
        "type_reward": 1.0,
        "content_reward": r0,
        "is_correct": r0 > 0.8,
        "pred_type": "click",
        "pred_action": {"function": "click", "coordinate": [int(pred_x), int(pred_y)]},
        "temperature": 0.0,
    })
    rewards.append(r0)

    # Candidates 1..N-1: perturbed
    for _ in range(n_candidates - 1):
        dx = rng.normal(0, coord_sigma)
        dy = rng.normal(0, coord_sigma)
        new_x = max(0, min(int(pred_x + dx), img_w))
        new_y = max(0, min(int(pred_y + dy), img_h))

        new_resp = _replace_coordinate(greedy_cand["response"], [new_x, new_y])
        cr = _gaussian_coord_reward([new_x, new_y], gt_coord, reward_sigma, img_w, img_h)

        candidates.append({
            "response": new_resp,
            "reward": 0.3 + 0.7 * cr,
            "type_reward": 1.0,
            "content_reward": cr,
            "is_correct": cr > 0.8,
            "pred_type": "click",
            "pred_action": {"function": "click", "coordinate": [new_x, new_y]},
            "temperature": 0.0,
        })
        rewards.append(cr)

    return candidates, rewards


# ═══════════════════════════════════════════════════════════════════════
# Type: text perturbation
# ═══════════════════════════════════════════════════════════════════════

def _levenshtein(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,       # deletion
                curr[j] + 1,            # insertion
                prev[j] + (c1 != c2),   # substitution
            ))
        prev = curr
    return prev[-1]


def _text_reward(pred_text, gt_text):
    """Continuous reward based on normalized edit distance.

    Returns 1.0 for exact match, 0.0 for completely different.
    """
    if pred_text == gt_text:
        return 1.0
    if not gt_text and not pred_text:
        return 1.0
    dist = _levenshtein(pred_text, gt_text)
    max_len = max(len(pred_text), len(gt_text), 1)
    return max(0.0, 1.0 - dist / max_len)


_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-_/@#"


def _perturb_text_string(text, rng):
    """Apply a random character-level perturbation to text."""
    if not text:
        return rng.choice(list(_CHARSET))

    op = rng.choice(["sub", "del", "ins", "trunc", "case", "multi"])
    t = text

    if op == "sub":
        # Substitute 1-2 random characters
        n_sub = min(rng.randint(1, 3), len(t))
        positions = rng.choice(len(t), n_sub, replace=False)
        t_list = list(t)
        for p in positions:
            t_list[p] = rng.choice(list(_CHARSET))
        t = "".join(t_list)
    elif op == "del" and len(t) > 1:
        # Delete 1 random character
        pos = rng.randint(0, len(t))
        t = t[:pos] + t[pos + 1:]
    elif op == "ins":
        # Insert 1 random character
        pos = rng.randint(0, len(t) + 1)
        t = t[:pos] + rng.choice(list(_CHARSET)) + t[pos:]
    elif op == "trunc" and len(t) > 1:
        # Truncate to 30-80% of original
        k = max(1, int(len(t) * rng.uniform(0.3, 0.8)))
        t = t[:k]
    elif op == "case":
        # Change case
        t = t.swapcase() if rng.random() < 0.5 else t.lower()
    elif op == "multi":
        # Multiple edits for higher edit distance
        n_edits = rng.randint(2, max(3, len(t) // 2 + 1))
        t_list = list(t)
        for _ in range(n_edits):
            if t_list:
                pos = rng.randint(0, len(t_list))
                t_list[pos] = rng.choice(list(_CHARSET))
        t = "".join(t_list)

    # Ensure perturbation is different
    if t == text:
        t = text + rng.choice(list(_CHARSET))
    return t


def _replace_text_in_response(response_text, new_text):
    """Replace the 'text' value in the <tool_call> block."""
    tc_start = response_text.find("<tool_call>")
    if tc_start < 0:
        return response_text
    before = response_text[:tc_start]
    after = response_text[tc_start:]

    # Escape the new text for JSON (includes surrounding quotes)
    escaped = json.dumps(new_text)

    # Match "text": "..." (handling escaped characters in value)
    # Use function replacement to avoid backslash interpretation issues
    pattern = r'"text"\s*:\s*"(?:[^"\\]|\\.)*"'

    def _replacer(_m):
        return f'"text": {escaped}'

    return before + re.sub(pattern, _replacer, after, count=1)


def _perturb_type(greedy_cand, gt_text, n_candidates, rng):
    """Generate N type candidates via text perturbation."""
    pred_action = _parse_stored(greedy_cand.get("pred_action", {}))
    pred_text = pred_action.get("text", "")

    candidates = []
    rewards = []

    # Candidate 0: original greedy
    r0 = _text_reward(pred_text, gt_text)
    candidates.append({
        "response": greedy_cand["response"],
        "reward": 0.3 + 0.7 * r0,
        "type_reward": 1.0,
        "content_reward": r0,
        "is_correct": r0 > 0.8,
        "pred_type": "type",
        "pred_action": {"function": "type", "text": pred_text},
        "temperature": 0.0,
    })
    rewards.append(r0)

    # Generate unique perturbations
    seen = {pred_text}
    for _ in range(n_candidates - 1):
        # Try a few times to get unique perturbation
        for _attempt in range(10):
            perturbed = _perturb_text_string(pred_text, rng)
            if perturbed not in seen:
                break
        seen.add(perturbed)

        new_resp = _replace_text_in_response(greedy_cand["response"], perturbed)
        cr = _text_reward(perturbed, gt_text)

        candidates.append({
            "response": new_resp,
            "reward": 0.3 + 0.7 * cr,
            "type_reward": 1.0,
            "content_reward": cr,
            "is_correct": cr > 0.8,
            "pred_type": "type",
            "pred_action": {"function": "type", "text": perturbed},
            "temperature": 0.0,
        })
        rewards.append(cr)

    return candidates, rewards


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Action-Space GRPO data preparation")
    parser.add_argument("--input", required=True,
                        help="Input GRPO candidates JSONL")
    parser.add_argument("--output", required=True,
                        help="Output AS-GRPO candidates JSONL")
    parser.add_argument("--n_candidates", type=int, default=8,
                        help="Candidates per step (1 original + N-1 perturbed)")
    parser.add_argument("--coord_sigma", type=float, default=40.0,
                        help="Click coordinate perturbation sigma (pixels)")
    parser.add_argument("--reward_sigma", type=float, default=0.05,
                        help="Click reward Gaussian sigma (normalized coords)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    logger.info(f"Action-Space GRPO data preparation")
    logger.info(f"  Input:          {args.input}")
    logger.info(f"  Output:         {args.output}")
    logger.info(f"  N candidates:   {args.n_candidates}")
    logger.info(f"  Coord sigma:    {args.coord_sigma} px")
    logger.info(f"  Reward sigma:   {args.reward_sigma} (normalized)")
    logger.info(f"  Action types:   click (coord perturbation) + type (text perturbation)")

    stats = {
        "total_groups": 0,
        "click_perturbed": 0,
        "type_perturbed": 0,
        "skipped": 0,       # no valid perturbation possible
        "kept_original": 0,  # swipe or other, kept as-is
    }
    click_rewards = []
    type_rewards = []

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            group = json.loads(line)
            stats["total_groups"] += 1

            gt_action = _parse_stored(group.get("gt_action", {}))
            gt_type = gt_action.get("function") or gt_action.get("action")
            img_w = group.get("image_w", 1040)
            img_h = group.get("image_h", 736)
            greedy = group["candidates"][0]

            # ── Click: coordinate perturbation ──
            if gt_type == "click" and greedy.get("pred_type") == "click":
                gt_coord = gt_action.get("coordinate")
                if gt_coord is not None:
                    result = _perturb_click(
                        greedy, gt_coord, args.n_candidates,
                        args.coord_sigma, args.reward_sigma,
                        img_w, img_h, rng)
                    if result is not None:
                        cands, rews = result
                        group["candidates"] = cands
                        click_rewards.extend(rews)
                        stats["click_perturbed"] += 1
                        fout.write(json.dumps(group, ensure_ascii=False) + "\n")
                        continue

            # ── Type: text perturbation ──
            if gt_type == "type" and greedy.get("pred_type") == "type":
                gt_text = gt_action.get("text", "")
                if gt_text:  # skip empty GT text
                    result = _perturb_type(
                        greedy, gt_text, args.n_candidates, rng)
                    if result is not None:
                        cands, rews = result
                        group["candidates"] = cands
                        type_rewards.extend(rews)
                        stats["type_perturbed"] += 1
                        fout.write(json.dumps(group, ensure_ascii=False) + "\n")
                        continue

            # ── Swipe / other / skip: keep original ──
            stats["kept_original"] += 1
            fout.write(json.dumps(group, ensure_ascii=False) + "\n")

    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Action-Space GRPO Data Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  Total groups:     {stats['total_groups']}")
    logger.info(f"  Click perturbed:  {stats['click_perturbed']}")
    logger.info(f"  Type perturbed:   {stats['type_perturbed']}")
    logger.info(f"  Kept original:    {stats['kept_original']}")
    logger.info(f"  Total AS samples: {(stats['click_perturbed'] + stats['type_perturbed']) * args.n_candidates}")

    if click_rewards:
        cr = np.array(click_rewards)
        logger.info(f"\n  Click reward distribution (N={len(cr)}):")
        logger.info(f"    mean={cr.mean():.3f}  std={cr.std():.3f}  median={np.median(cr):.3f}")
        logger.info(f"    >0.8: {(cr > 0.8).mean()*100:.1f}%  >0.5: {(cr > 0.5).mean()*100:.1f}%  <0.01: {(cr < 0.01).mean()*100:.1f}%")
        # Per-group variance
        n = args.n_candidates
        gstd = cr[:len(cr) // n * n].reshape(-1, n).std(axis=1)
        logger.info(f"    Per-group std: mean={gstd.mean():.4f}  >0.01={gstd[gstd > 0.01].shape[0]} ({(gstd > 0.01).mean()*100:.1f}%)")

    if type_rewards:
        tr = np.array(type_rewards)
        logger.info(f"\n  Type reward distribution (N={len(tr)}):")
        logger.info(f"    mean={tr.mean():.3f}  std={tr.std():.3f}  median={np.median(tr):.3f}")
        logger.info(f"    >0.8: {(tr > 0.8).mean()*100:.1f}%  >0.5: {(tr > 0.5).mean()*100:.1f}%  <0.01: {(tr < 0.01).mean()*100:.1f}%")
        n = args.n_candidates
        gstd = tr[:len(tr) // n * n].reshape(-1, n).std(axis=1)
        logger.info(f"    Per-group std: mean={gstd.mean():.4f}  >0.01={gstd[gstd > 0.01].shape[0]} ({(gstd > 0.01).mean()*100:.1f}%)")

    logger.info(f"\n{'='*60}")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
