"""
Detailed analysis of evaluation results across methods and epochs.
Compares: V13 (epochs 0-3), V12 (epochs 0-3), Std LoRA+SP, Std LoRA+GRPO,
          Full-param+SP, Full-param+GRPO
"""
import json
import os
import sys
from collections import Counter, defaultdict

PROJECT = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

# ── Load test data for episode metadata ──
with open(f"{PROJECT}/v13_gui_360/data/gui360_test_968.jsonl") as f:
    test_episodes = [json.loads(l) for l in f]

# Build metadata: episode_id -> {app, num_steps, goal, length_bucket}
meta = {}
for ep in test_episodes:
    eid = ep["episode_id"]
    path = ep["steps"][0]["screenshot"]
    parts = path.split("/")
    app = parts[parts.index("image") + 1]
    ns = ep["num_steps"]
    bucket = "short" if ns <= 3 else ("medium" if ns <= 7 else "long")
    meta[eid] = {"app": app, "num_steps": ns, "goal": ep["goal"], "bucket": bucket}

# ── Define methods and their result paths ──
def find_latest(dirpath):
    """Find latest eval_results_*.json in a directory."""
    if not os.path.isdir(dirpath):
        return None
    files = [f for f in os.listdir(dirpath) if f.startswith("eval_results_") and f.endswith(".json")]
    if not files:
        return None
    files.sort()
    return os.path.join(dirpath, files[-1])

methods = {}
# V13 epochs 0-3
for ep in range(4):
    p = find_latest(f"{PROJECT}/v13_gui_360/outputs/epoch-{ep}")
    if p:
        methods[f"V13_SP_ep{ep}"] = p

# V13 resumed
for tag in ["3-resumed", "5-resumed"]:
    p = find_latest(f"{PROJECT}/v13_gui_360/outputs/epoch-{tag}")
    if p:
        methods[f"V13_SP_ep{tag}"] = p

# V12 epochs 0-3
for ep in range(4):
    p = find_latest(f"{PROJECT}/v12_gui_360/outputs/epoch-{ep}")
    if p:
        methods[f"V12_SP_ep{ep}"] = p

# Baselines
for label, subdir in [
    ("StdLoRA_SP", "std_lora_sp"),
    ("StdLoRA_GRPO", "std_lora_grpo"),
    ("FullParam_SP", "fullparam_sp"),
    ("FullParam_GRPO", "fullparam_grpo"),
    ("V12_GRPO", "v12_grpo"),
]:
    for ep in range(4):
        p = find_latest(f"{PROJECT}/v12_gui_360/outputs/{subdir}/epoch-{ep}")
        if p:
            methods[f"{label}_ep{ep}"] = p

# ── Load all results ──
results = {}
for name, path in methods.items():
    with open(path) as f:
        data = json.load(f)
    episodes = data["episodes"]
    # Build dict: episode_id -> episode result
    ep_dict = {}
    for ep in episodes:
        ep_dict[ep["episode_id"]] = ep
    results[name] = ep_dict

print(f"Loaded {len(results)} method-epoch combinations")
print("Methods:", list(results.keys()))
print()

# ── Helper functions ──
def compute_metrics(ep_dict, episode_ids=None):
    """Compute TSR, progress, step SR for a subset of episodes."""
    if episode_ids is None:
        episode_ids = list(ep_dict.keys())
    total = 0
    success = 0
    progress_sum = 0
    steps_correct = 0
    steps_total = 0
    for eid in episode_ids:
        if eid not in ep_dict:
            continue
        ep = ep_dict[eid]
        total += 1
        if ep["tsr"] > 0.99:
            success += 1
        progress_sum += ep["progress"]
        steps_correct += ep["correct_steps"]
        steps_total += ep["evaluated_steps"]
    if total == 0:
        return {"tsr": 0, "progress": 0, "step_sr": 0, "n": 0, "success": 0}
    return {
        "tsr": success / total,
        "progress": progress_sum / total,
        "step_sr": steps_correct / steps_total if steps_total > 0 else 0,
        "n": total,
        "success": success,
    }

# ── 1. Breakdown by App Type ──
print("=" * 80)
print("1. TSR BY APP TYPE (best epoch for each method)")
print("=" * 80)

# Focus on best epochs
best_methods = {
    "V13+SP ep3": "V13_SP_ep3",
    "V13+SP ep3-res": "V13_SP_ep3-resumed",
    "V12+SP ep3": "V12_SP_ep3",
    "StdLoRA+SP ep3": "StdLoRA_SP_ep3",
    "StdLoRA+GRPO ep3": "StdLoRA_GRPO_ep3",
    "FullParam+SP ep3": "FullParam_SP_ep3",
    "FullParam+GRPO ep3": "FullParam_GRPO_ep3",
}
# Add V12 GRPO if available
if "V12_GRPO_ep2" in results:
    best_methods["V12+GRPO ep2"] = "V12_GRPO_ep2"

apps = ["word", "excel", "ppt"]
app_ids = {app: [e["episode_id"] for e in test_episodes if meta[e["episode_id"]]["app"] == app] for app in apps}

header = f"{'Method':<22} | {'Word':>12} | {'Excel':>12} | {'PPT':>12} | {'Total':>12}"
print(header)
print("-" * len(header))
for label, key in best_methods.items():
    if key not in results:
        continue
    ep_dict = results[key]
    parts = []
    for app in apps:
        m = compute_metrics(ep_dict, app_ids[app])
        parts.append(f"{m['tsr']*100:5.1f}% ({m['success']:>3}/{m['n']})")
    total = compute_metrics(ep_dict)
    parts.append(f"{total['tsr']*100:5.1f}% ({total['success']:>3}/{total['n']})")
    print(f"{label:<22} | {'|'.join(f'{p:>12}' for p in parts)}")

# ── 2. Breakdown by Trajectory Length ──
print()
print("=" * 80)
print("2. TSR BY TRAJECTORY LENGTH")
print("=" * 80)

buckets = ["short", "medium", "long"]
bucket_labels = {"short": "Short(1-3)", "medium": "Med(4-7)", "long": "Long(8+)"}
bucket_ids = {b: [e["episode_id"] for e in test_episodes if meta[e["episode_id"]]["bucket"] == b] for b in buckets}

header = f"{'Method':<22} | {'Short(1-3)':>14} | {'Med(4-7)':>14} | {'Long(8+)':>14} | {'Total':>14}"
print(header)
print("-" * len(header))
for label, key in best_methods.items():
    if key not in results:
        continue
    ep_dict = results[key]
    parts = []
    for b in buckets:
        m = compute_metrics(ep_dict, bucket_ids[b])
        parts.append(f"{m['tsr']*100:5.1f}% ({m['success']:>3}/{m['n']})")
    total = compute_metrics(ep_dict)
    parts.append(f"{total['tsr']*100:5.1f}% ({total['success']:>3}/{total['n']})")
    print(f"{label:<22} | {'|'.join(f'{p:>14}' for p in parts)}")

# ── 3. V13 epoch-to-epoch gain analysis ──
print()
print("=" * 80)
print("3. V13 EPOCH-BY-EPOCH GAIN: WHAT TRAJECTORIES DID EACH EPOCH FIX?")
print("=" * 80)

v13_epochs = ["V13_SP_ep0", "V13_SP_ep1", "V13_SP_ep2", "V13_SP_ep3"]
available_v13 = [e for e in v13_epochs if e in results]

for i in range(1, len(available_v13)):
    prev_key = available_v13[i-1]
    curr_key = available_v13[i]
    prev = results[prev_key]
    curr = results[curr_key]

    # Find newly correct and newly broken trajectories
    all_ids = set(prev.keys()) & set(curr.keys())
    gained = []  # was wrong, now correct
    lost = []    # was correct, now wrong
    for eid in all_ids:
        prev_ok = prev[eid]["tsr"] > 0.99
        curr_ok = curr[eid]["tsr"] > 0.99
        if not prev_ok and curr_ok:
            gained.append(eid)
        elif prev_ok and not curr_ok:
            lost.append(eid)

    print(f"\n{prev_key} → {curr_key}: +{len(gained)} gained, -{len(lost)} lost (net {len(gained)-len(lost):+d})")

    # Breakdown of gained by app and length
    gained_by_app = Counter(meta[eid]["app"] for eid in gained)
    gained_by_bucket = Counter(meta[eid]["bucket"] for eid in gained)
    lost_by_app = Counter(meta[eid]["app"] for eid in lost)
    lost_by_bucket = Counter(meta[eid]["bucket"] for eid in lost)

    print(f"  Gained by app:    word={gained_by_app['word']}, excel={gained_by_app['excel']}, ppt={gained_by_app['ppt']}")
    print(f"  Lost by app:      word={lost_by_app['word']}, excel={lost_by_app['excel']}, ppt={lost_by_app['ppt']}")
    print(f"  Gained by length: short={gained_by_bucket['short']}, med={gained_by_bucket['medium']}, long={gained_by_bucket['long']}")
    print(f"  Lost by length:   short={lost_by_bucket['short']}, med={lost_by_bucket['medium']}, long={lost_by_bucket['long']}")

    # Step count distribution of gained
    gained_steps = Counter(meta[eid]["num_steps"] for eid in gained)
    print(f"  Gained step distribution: {dict(sorted(gained_steps.items()))}")

# ── 4. Cross-method comparison: Who solves what uniquely? ──
print()
print("=" * 80)
print("4. CROSS-METHOD UNIQUE SOLVES (best epoch per method)")
print("=" * 80)

method_solved = {}
for label, key in best_methods.items():
    if key not in results:
        continue
    solved = set()
    for eid, ep in results[key].items():
        if ep["tsr"] > 0.99:
            solved.add(eid)
    method_solved[label] = solved

# Union of all solved
all_solved = set()
for s in method_solved.values():
    all_solved |= s
print(f"Total unique episodes solved by ANY method: {len(all_solved)}/{len(test_episodes)}")

# For each method, unique solves
for label, solved in method_solved.items():
    others = set()
    for other_label, other_solved in method_solved.items():
        if other_label != label:
            others |= other_solved
    unique = solved - others
    print(f"  {label:<22}: solved={len(solved):>3}, unique={len(unique):>2}")
    if unique:
        apps = Counter(meta[eid]["app"] for eid in unique)
        buckets = Counter(meta[eid]["bucket"] for eid in unique)
        print(f"    Unique by app: {dict(apps)}, by length: {dict(buckets)}")

# ── 5. V13 vs V12 vs StdLoRA detailed comparison ──
print()
print("=" * 80)
print("5. V13 ep3 vs V12 ep3 vs StdLoRA+GRPO ep3: HEAD-TO-HEAD")
print("=" * 80)

compare_keys = {
    "V13+SP": "V13_SP_ep3",
    "V12+SP": "V12_SP_ep3",
    "StdLoRA+GRPO": "StdLoRA_GRPO_ep3",
}
compare_sets = {}
for label, key in compare_keys.items():
    if key not in results:
        continue
    compare_sets[label] = set(eid for eid, ep in results[key].items() if ep["tsr"] > 0.99)

labels = list(compare_sets.keys())
for i, l1 in enumerate(labels):
    for j, l2 in enumerate(labels):
        if i >= j:
            continue
        only_1 = compare_sets[l1] - compare_sets[l2]
        only_2 = compare_sets[l2] - compare_sets[l1]
        both = compare_sets[l1] & compare_sets[l2]
        print(f"\n{l1} vs {l2}:")
        print(f"  Both solve:     {len(both)}")
        print(f"  Only {l1:<15}: {len(only_1)}")
        print(f"  Only {l2:<15}: {len(only_2)}")

        # Analyze what's unique to each
        for tag, unique_set in [(l1, only_1), (l2, only_2)]:
            if unique_set:
                apps = Counter(meta[eid]["app"] for eid in unique_set)
                buckets = Counter(meta[eid]["bucket"] for eid in unique_set)
                steps = [meta[eid]["num_steps"] for eid in unique_set]
                avg_steps = sum(steps) / len(steps)
                print(f"    Only-{tag}: apps={dict(apps)}, lengths={dict(buckets)}, avg_steps={avg_steps:.1f}")

# ── 6. Error type analysis ──
print()
print("=" * 80)
print("6. FIRST-ERROR ANALYSIS (where do trajectories fail?)")
print("=" * 80)

for label, key in [("V13+SP ep3", "V13_SP_ep3"), ("V12+SP ep3", "V12_SP_ep3"),
                     ("StdLoRA+GRPO ep3", "StdLoRA_GRPO_ep3"), ("StdLoRA+SP ep3", "StdLoRA_SP_ep3")]:
    if key not in results:
        continue
    ep_dict = results[key]

    # For failed trajectories, analyze first error
    type_mismatch = 0  # wrong action type
    content_mismatch = 0  # right type, wrong content
    format_error = 0  # format error
    total_failed = 0
    first_error_step_dist = Counter()

    for eid, ep in ep_dict.items():
        if ep["tsr"] > 0.99:
            continue
        total_failed += 1

        # First error step
        fe = ep.get("first_error_step", 1)
        ns = meta[eid]["num_steps"]
        # Normalize to fraction of trajectory
        frac = fe / ns if ns > 0 else 0
        if frac <= 0.25:
            first_error_step_dist["0-25%"] += 1
        elif frac <= 0.5:
            first_error_step_dist["25-50%"] += 1
        elif frac <= 0.75:
            first_error_step_dist["50-75%"] += 1
        else:
            first_error_step_dist["75-100%"] += 1

        # Check first error type
        if ep["step_results"]:
            for sr in ep["step_results"]:
                if not sr["success"]:
                    if sr.get("format_reward", 1) < 0.5:
                        format_error += 1
                    elif sr.get("type_reward", 1) < 0.5:
                        type_mismatch += 1
                    else:
                        content_mismatch += 1
                    break

    print(f"\n{label} ({total_failed} failed trajectories):")
    print(f"  First error type: format={format_error}, type_mismatch={type_mismatch}, content_wrong={content_mismatch}")
    print(f"  First error position: {dict(sorted(first_error_step_dist.items()))}")

# ── 7. Progress distribution ──
print()
print("=" * 80)
print("7. PROGRESS DISTRIBUTION (how far do failed trajectories get?)")
print("=" * 80)

for label, key in [("V13+SP ep3", "V13_SP_ep3"), ("V12+SP ep3", "V12_SP_ep3"),
                     ("StdLoRA+GRPO ep3", "StdLoRA_GRPO_ep3")]:
    if key not in results:
        continue
    ep_dict = results[key]

    progress_bins = Counter()
    for eid, ep in ep_dict.items():
        if ep["tsr"] > 0.99:
            progress_bins["100%"] += 1
            continue
        p = ep["progress"]
        if p == 0:
            progress_bins["0%"] += 1
        elif p <= 0.25:
            progress_bins["1-25%"] += 1
        elif p <= 0.5:
            progress_bins["26-50%"] += 1
        elif p <= 0.75:
            progress_bins["51-75%"] += 1
        else:
            progress_bins["76-99%"] += 1

    print(f"\n{label}:")
    for b in ["0%", "1-25%", "26-50%", "51-75%", "76-99%", "100%"]:
        cnt = progress_bins.get(b, 0)
        bar = "#" * (cnt // 5)
        print(f"  {b:>6}: {cnt:>4} {bar}")

# ── 8. Step-level action type analysis ──
print()
print("=" * 80)
print("8. STEP-LEVEL ACTION TYPE SUCCESS RATES")
print("=" * 80)

for label, key in [("V13+SP ep3", "V13_SP_ep3"), ("V12+SP ep3", "V12_SP_ep3"),
                     ("StdLoRA+GRPO ep3", "StdLoRA_GRPO_ep3")]:
    if key not in results:
        continue
    ep_dict = results[key]

    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for eid, ep in ep_dict.items():
        for sr in ep["step_results"]:
            gt_type = sr.get("gt_type", "unknown")
            type_stats[gt_type]["total"] += 1
            if sr["success"]:
                type_stats[gt_type]["correct"] += 1

    print(f"\n{label}:")
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        rate = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {t:<15}: {rate:5.1f}% ({s['correct']}/{s['total']})")
