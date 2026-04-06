"""Q5: Pattern Memorization vs Compositional Generalization

Is the ~20pp planning error from pattern memorization or compositional failure?

Analyses:
  1. App extraction from goal text
  2. Bigram novelty score per episode
  3. Novelty-binned accuracy (baseline, oracle, gap)
  4. Difficulty-controlled analysis (within oracle-accuracy bins, split by novelty)
  5. App-frequency analysis (rare apps → lower accuracy?)
  6. Action type diversity analysis

Data sources:
  - dataset: datasets/android_control_evaluation_std.jsonl (1543 episodes)
  - baseline: outputs/eval_a_ac/.../trajectory_results.jsonl
  - oracle:   outputs/eval_context_subtask/.../trajectory_results.jsonl
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict, Counter

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

ALL_ACTION_TYPES = ['click', 'long_press', 'swipe', 'type', 'open', 'system_button', 'wait']


def save_json(data, path):
    """Save dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _default(obj):
        if hasattr(obj, 'item'): return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)


def load_trajectory_results(path):
    """Load trajectory results indexed by episode_id."""
    results = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line.strip())
            results[ep['episode_id']] = ep
    return results


def load_dataset(path):
    """Load dataset episodes indexed by episode_id."""
    episodes = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line.strip())
            episodes[ep['episode_id']] = ep
    return episodes


def extract_app_from_goal(goal):
    """Extract app name from goal text using regex patterns."""
    goal_lower = goal.lower()

    # Pattern 1: "on the X app" / "in the X app"
    m = re.search(r'(?:on|in|using|open|from|via|through)\s+(?:the\s+)?([A-Z][A-Za-z0-9\s\-\.]+?)\s+app', goal, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()

    # Pattern 2: "open X" at start
    m = re.search(r'^open\s+(?:the\s+)?([A-Z][A-Za-z0-9\s\-\.]+?)(?:\s+and|\s+to|\s*$|\.)', goal, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()

    # Pattern 3: Known app names anywhere in text
    known_apps = [
        'chrome', 'settings', 'gmail', 'maps', 'youtube', 'camera', 'clock',
        'calendar', 'contacts', 'messages', 'phone', 'photos', 'play store',
        'google play', 'calculator', 'files', 'notes', 'weather', 'spotify',
        'whatsapp', 'instagram', 'facebook', 'twitter', 'snapchat', 'tiktok',
        'amazon', 'ebay', 'netflix', 'uber', 'lyft', 'airbnb', 'yelp',
        'reddit', 'pinterest', 'linkedin', 'telegram', 'discord', 'zoom',
        'teams', 'slack', 'notion', 'todoist', 'trello', 'evernote',
        'snapdeal', 'flipkart', 'myntra', 'paytm', 'swiggy', 'zomato',
        'booking.com', 'trivago', 'expedia', 'skyscanner', 'kayak',
        'duolingo', 'kindle', 'audible', 'shazam', 'soundcloud',
        'gallery', 'browser', 'dialer', 'launcher', 'store',
    ]
    for app in known_apps:
        if app in goal_lower:
            return app

    # Pattern 4: "X app" anywhere
    m = re.search(r'([A-Z][A-Za-z0-9]+)\s+app', goal)
    if m:
        return m.group(1).strip().lower()

    return 'unknown'


def compute_action_bigrams(action_types):
    """Compute action type bigrams from a sequence."""
    bigrams = []
    for i in range(len(action_types) - 1):
        bigrams.append((action_types[i], action_types[i + 1]))
    return bigrams


def compute_step_accuracy(ep_result):
    """Compute step-level accuracy for an episode result."""
    steps = ep_result.get('step_results', [])
    if not steps:
        return 0.0
    correct = sum(1 for s in steps if s.get('extract_match', False))
    return correct / len(steps)


def compute_type_accuracy(ep_result):
    """Compute action type accuracy for an episode result."""
    steps = ep_result.get('step_results', [])
    if not steps:
        return 0.0
    correct = sum(1 for s in steps if s.get('type_match', False))
    return correct / len(steps)


def main():
    parser = argparse.ArgumentParser(description="Q5: Compositional Generalization Analysis")
    parser.add_argument("--baseline_results", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'eval_a_ac', 'Qwen2.5-VL-7B', 'trajectory_results.jsonl'))
    parser.add_argument("--oracle_results", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'eval_context_subtask', 'Qwen2.5-VL-7B', 'trajectory_results.jsonl'))
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(PROJECT_ROOT, 'datasets', 'android_control_evaluation_std.jsonl'))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'analysis_q5_compositional'))
    args = parser.parse_args()

    print("Loading data...")
    dataset = load_dataset(args.dataset)
    baseline_results = load_trajectory_results(args.baseline_results)
    oracle_results = load_trajectory_results(args.oracle_results)
    print(f"Dataset: {len(dataset)} episodes, Baseline: {len(baseline_results)}, Oracle: {len(oracle_results)}")

    # --- Analysis 1: App extraction ---
    app_counts = Counter()
    episode_apps = {}
    for eid, ep in dataset.items():
        app = extract_app_from_goal(ep['goal'])
        app_counts[app] += 1
        episode_apps[eid] = app

    known_count = sum(v for k, v in app_counts.items() if k != 'unknown')
    print(f"\nApp extraction: {known_count}/{len(dataset)} episodes matched ({known_count/len(dataset)*100:.1f}%)")
    print(f"Unique apps: {len(app_counts)}")
    print(f"Top 10 apps: {app_counts.most_common(10)}")

    # --- Analysis 2: Bigram novelty score ---
    # First compute global bigram frequencies from all episodes
    global_bigrams = Counter()
    episode_bigrams = {}
    episode_action_types = {}

    for eid, ep in dataset.items():
        types = [s['action_content']['action'] for s in ep['steps']]
        episode_action_types[eid] = types
        bgs = compute_action_bigrams(types)
        episode_bigrams[eid] = bgs
        for bg in bgs:
            global_bigrams[bg] += 1

    total_bigrams = sum(global_bigrams.values())
    print(f"\nTotal unique bigrams: {len(global_bigrams)}, total count: {total_bigrams}")

    # Novelty score = average inverse frequency of bigrams
    episode_novelty = {}
    for eid, bgs in episode_bigrams.items():
        if not bgs:
            episode_novelty[eid] = 0.0
            continue
        inv_freqs = []
        for bg in bgs:
            freq = global_bigrams[bg] / total_bigrams
            inv_freqs.append(-math.log(freq + 1e-10))  # log-inverse frequency
        episode_novelty[eid] = float(np.mean(inv_freqs))

    novelty_values = list(episode_novelty.values())
    print(f"Novelty score: mean={np.mean(novelty_values):.3f}, std={np.std(novelty_values):.3f}, "
          f"min={np.min(novelty_values):.3f}, max={np.max(novelty_values):.3f}")

    # --- Analysis 3: Novelty-binned accuracy ---
    # Split episodes into quintiles by novelty
    sorted_eids = sorted(episode_novelty.keys(), key=lambda x: episode_novelty[x])
    n = len(sorted_eids)
    quintile_size = n // 5

    novelty_bins = {}
    for q in range(5):
        start = q * quintile_size
        end = (q + 1) * quintile_size if q < 4 else n
        bin_eids = sorted_eids[start:end]
        bin_label = f"Q{q+1}"

        bin_novelty = [episode_novelty[e] for e in bin_eids]
        bin_baseline_acc = []
        bin_oracle_acc = []
        bin_baseline_type_acc = []
        bin_oracle_type_acc = []

        for eid in bin_eids:
            if eid in baseline_results:
                bin_baseline_acc.append(compute_step_accuracy(baseline_results[eid]))
                bin_baseline_type_acc.append(compute_type_accuracy(baseline_results[eid]))
            if eid in oracle_results:
                bin_oracle_acc.append(compute_step_accuracy(oracle_results[eid]))
                bin_oracle_type_acc.append(compute_type_accuracy(oracle_results[eid]))

        bl_mean = float(np.mean(bin_baseline_acc)) if bin_baseline_acc else 0
        oc_mean = float(np.mean(bin_oracle_acc)) if bin_oracle_acc else 0
        bl_type_mean = float(np.mean(bin_baseline_type_acc)) if bin_baseline_type_acc else 0
        oc_type_mean = float(np.mean(bin_oracle_type_acc)) if bin_oracle_type_acc else 0

        novelty_bins[bin_label] = {
            'n_episodes': len(bin_eids),
            'novelty_range': [float(np.min(bin_novelty)), float(np.max(bin_novelty))],
            'novelty_mean': float(np.mean(bin_novelty)),
            'baseline_step_acc': bl_mean,
            'oracle_step_acc': oc_mean,
            'gap': oc_mean - bl_mean,
            'baseline_type_acc': bl_type_mean,
            'oracle_type_acc': oc_type_mean,
            'type_gap': oc_type_mean - bl_type_mean,
            'baseline_tsr': sum(1 for e in bin_eids if baseline_results.get(e, {}).get('task_success', False)) / len(bin_eids),
            'oracle_tsr': sum(1 for e in bin_eids if oracle_results.get(e, {}).get('task_success', False)) / len(bin_eids),
        }

    print("\nNovelty-binned accuracy:")
    for q, d in novelty_bins.items():
        print(f"  {q}: novelty={d['novelty_mean']:.2f}, bl_acc={d['baseline_step_acc']:.3f}, "
              f"oc_acc={d['oracle_step_acc']:.3f}, gap={d['gap']:.3f}, n={d['n_episodes']}")

    # --- Analysis 4: Difficulty-controlled analysis ---
    # Within same oracle-accuracy bins, further split by novelty
    difficulty_controlled = {}
    # Bin episodes by oracle type accuracy into tertiles
    oracle_type_accs = {}
    for eid in dataset:
        if eid in oracle_results:
            oracle_type_accs[eid] = compute_type_accuracy(oracle_results[eid])

    sorted_by_oracle = sorted(oracle_type_accs.keys(), key=lambda x: oracle_type_accs[x])
    tertile_size = len(sorted_by_oracle) // 3

    for t in range(3):
        start = t * tertile_size
        end = (t + 1) * tertile_size if t < 2 else len(sorted_by_oracle)
        tertile_eids = sorted_by_oracle[start:end]
        tertile_label = ['low_oracle', 'mid_oracle', 'high_oracle'][t]

        # Split this tertile by novelty median
        tertile_novelties = [(e, episode_novelty.get(e, 0)) for e in tertile_eids]
        median_novelty = float(np.median([n for _, n in tertile_novelties]))

        low_novelty = [e for e, n in tertile_novelties if n <= median_novelty]
        high_novelty = [e for e, n in tertile_novelties if n > median_novelty]

        def compute_group_stats(eids):
            bl_accs = [compute_step_accuracy(baseline_results[e]) for e in eids if e in baseline_results]
            oc_accs = [compute_step_accuracy(oracle_results[e]) for e in eids if e in oracle_results]
            bl_type = [compute_type_accuracy(baseline_results[e]) for e in eids if e in baseline_results]
            oc_type = [compute_type_accuracy(oracle_results[e]) for e in eids if e in oracle_results]
            return {
                'n': len(eids),
                'baseline_step_acc': float(np.mean(bl_accs)) if bl_accs else 0,
                'oracle_step_acc': float(np.mean(oc_accs)) if oc_accs else 0,
                'baseline_type_acc': float(np.mean(bl_type)) if bl_type else 0,
                'oracle_type_acc': float(np.mean(oc_type)) if oc_type else 0,
                'gap': (float(np.mean(oc_accs)) - float(np.mean(bl_accs))) if bl_accs and oc_accs else 0,
            }

        difficulty_controlled[tertile_label] = {
            'oracle_acc_range': [float(oracle_type_accs[tertile_eids[0]]),
                                 float(oracle_type_accs[tertile_eids[-1]])],
            'low_novelty': compute_group_stats(low_novelty),
            'high_novelty': compute_group_stats(high_novelty),
            'novelty_median': median_novelty,
        }

    print("\nDifficulty-controlled analysis:")
    for t, d in difficulty_controlled.items():
        lo = d['low_novelty']
        hi = d['high_novelty']
        print(f"  {t} (oracle range {d['oracle_acc_range'][0]:.2f}-{d['oracle_acc_range'][1]:.2f}):")
        print(f"    Low novelty:  bl={lo['baseline_step_acc']:.3f}, oc={lo['oracle_step_acc']:.3f}, gap={lo['gap']:.3f} (n={lo['n']})")
        print(f"    High novelty: bl={hi['baseline_step_acc']:.3f}, oc={hi['oracle_step_acc']:.3f}, gap={hi['gap']:.3f} (n={hi['n']})")

    # --- Analysis 5: App-frequency analysis ---
    app_stats = {}
    for app, count in app_counts.most_common():
        if app == 'unknown' or count < 5:
            continue
        app_eids = [e for e, a in episode_apps.items() if a == app]
        bl_accs = [compute_step_accuracy(baseline_results[e]) for e in app_eids if e in baseline_results]
        oc_accs = [compute_step_accuracy(oracle_results[e]) for e in app_eids if e in oracle_results]
        app_stats[app] = {
            'count': count,
            'baseline_step_acc': float(np.mean(bl_accs)) if bl_accs else 0,
            'oracle_step_acc': float(np.mean(oc_accs)) if oc_accs else 0,
            'gap': (float(np.mean(oc_accs)) - float(np.mean(bl_accs))) if bl_accs and oc_accs else 0,
        }

    # Correlation between app frequency and baseline accuracy
    if app_stats:
        freqs = [d['count'] for d in app_stats.values()]
        accs = [d['baseline_step_acc'] for d in app_stats.values()]
        if len(freqs) > 2:
            corr = float(np.corrcoef(np.log(freqs), accs)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    print(f"\nApp-frequency correlation (log_freq vs baseline_acc): r={corr:.3f}")

    # --- Analysis 6: Action type diversity ---
    diversity_stats = defaultdict(lambda: {'baseline_accs': [], 'oracle_accs': [], 'eids': []})
    for eid, types in episode_action_types.items():
        n_unique = len(set(types))
        diversity_stats[n_unique]['eids'].append(eid)
        if eid in baseline_results:
            diversity_stats[n_unique]['baseline_accs'].append(compute_step_accuracy(baseline_results[eid]))
        if eid in oracle_results:
            diversity_stats[n_unique]['oracle_accs'].append(compute_step_accuracy(oracle_results[eid]))

    diversity_results = {}
    for n_unique in sorted(diversity_stats.keys()):
        d = diversity_stats[n_unique]
        bl = d['baseline_accs']
        oc = d['oracle_accs']
        diversity_results[str(n_unique)] = {
            'n_episodes': len(d['eids']),
            'baseline_step_acc': float(np.mean(bl)) if bl else 0,
            'oracle_step_acc': float(np.mean(oc)) if oc else 0,
            'gap': (float(np.mean(oc)) - float(np.mean(bl))) if bl and oc else 0,
        }

    print("\nAction type diversity:")
    for nu, d in diversity_results.items():
        print(f"  {nu} unique types: bl={d['baseline_step_acc']:.3f}, oc={d['oracle_step_acc']:.3f}, gap={d['gap']:.3f} (n={d['n_episodes']})")

    # Assemble output
    output = {
        'app_extraction': {
            'coverage': known_count / len(dataset),
            'n_matched': known_count,
            'n_total': len(dataset),
            'n_unique_apps': len(app_counts),
            'top_apps': dict(app_counts.most_common(20)),
        },
        'bigram_novelty': {
            'n_unique_bigrams': len(global_bigrams),
            'total_bigrams': total_bigrams,
            'novelty_stats': {
                'mean': float(np.mean(novelty_values)),
                'std': float(np.std(novelty_values)),
                'min': float(np.min(novelty_values)),
                'max': float(np.max(novelty_values)),
            },
        },
        'novelty_binned_accuracy': novelty_bins,
        'difficulty_controlled': difficulty_controlled,
        'app_frequency': {
            'correlation_log_freq_vs_acc': corr,
            'per_app': app_stats,
        },
        'action_type_diversity': diversity_results,
        'summary': {
            'novelty_gap_trend': (
                f"Q1 gap={novelty_bins['Q1']['gap']:.3f}, "
                f"Q5 gap={novelty_bins['Q5']['gap']:.3f}. "
                f"{'Increasing' if novelty_bins['Q5']['gap'] > novelty_bins['Q1']['gap'] else 'Decreasing'} "
                f"gap with novelty suggests "
                f"{'compositional failure' if novelty_bins['Q5']['gap'] > novelty_bins['Q1']['gap'] else 'memorization bottleneck'}."
            ),
            'app_freq_correlation': corr,
        },
    }

    save_json(output, os.path.join(args.output_dir, 'q5_results.json'))
    print(f"\nResults saved to {os.path.join(args.output_dir, 'q5_results.json')}")


if __name__ == "__main__":
    main()
