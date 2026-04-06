"""Q5: Pattern Memorization vs Compositional Generalization — GUI-360 Version

Adapts AC Q5 analysis to GUI-360 dataset.
Key differences from AC:
  - Domain extraction: from episode_id (excel_*/word_*/ppt_*) instead of app regex
  - ~16 action types vs AC 7
  - SFT vs baseline comparison replaces oracle vs baseline
  - Flat step list data format

Data sources:
  - baseline: outputs/gui360_eval_results/baseline.json
  - sft:      outputs/gui360_eval_results/sft.json
  - dataset:  datasets/GUI-360/rl_data/gui360_test.jsonl
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict, Counter

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _default(obj):
        if hasattr(obj, 'item'): return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)


def load_eval_results(path):
    """Load GUI-360 eval results, return per-episode step lists and metadata."""
    with open(path) as f:
        data = json.load(f)
    episodes = defaultdict(list)
    for r in data['results']:
        episodes[r['episode_id']].append(r)
    for eid in episodes:
        episodes[eid].sort(key=lambda x: x['step_num'])
    return dict(episodes), data


def load_dataset(path):
    """Load gui360_test.jsonl indexed by execution_id."""
    episodes = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line.strip())
            episodes[ep['execution_id']] = ep
    return episodes


def extract_domain(episode_id):
    """Extract domain (excel/word/ppt) from episode_id prefix."""
    eid_lower = episode_id.lower()
    if eid_lower.startswith('excel'):
        return 'excel'
    elif eid_lower.startswith('word'):
        return 'word'
    elif eid_lower.startswith('ppt') or eid_lower.startswith('powerpoint'):
        return 'ppt'
    return 'unknown'


def compute_action_bigrams(action_types):
    bigrams = []
    for i in range(len(action_types) - 1):
        bigrams.append((action_types[i], action_types[i + 1]))
    return bigrams


def compute_step_accuracy(steps):
    """Compute step-level extract_match accuracy for a list of step results."""
    if not steps:
        return 0.0
    correct = sum(1 for s in steps if s.get('extract_match', False))
    return correct / len(steps)


def compute_type_accuracy(steps):
    """Compute action type accuracy for a list of step results."""
    if not steps:
        return 0.0
    correct = sum(1 for s in steps if s.get('type_match', False))
    return correct / len(steps)


def main():
    parser = argparse.ArgumentParser(description="Q5: Compositional Generalization — GUI-360")
    parser.add_argument("--baseline_results", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'gui360_eval_results', 'baseline.json'))
    parser.add_argument("--sft_results", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'gui360_eval_results', 'sft.json'))
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(PROJECT_ROOT, 'datasets', 'GUI-360', 'rl_data', 'gui360_train.jsonl'))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'analysis_q5_compositional_gui360'))
    args = parser.parse_args()

    print("Loading data...")
    dataset = load_dataset(args.dataset)
    baseline_eps, baseline_meta = load_eval_results(args.baseline_results)
    sft_eps, sft_meta = load_eval_results(args.sft_results)
    print(f"Dataset: {len(dataset)} episodes, Baseline: {len(baseline_eps)}, SFT: {len(sft_eps)}")

    # --- Analysis 1: Domain extraction ---
    domain_counts = Counter()
    episode_domains = {}
    for eid in dataset:
        domain = extract_domain(eid)
        domain_counts[domain] += 1
        episode_domains[eid] = domain

    # Also extract domains for eval episodes
    eval_domains = {}
    for eid in baseline_eps:
        eval_domains[eid] = extract_domain(eid)

    known_count = sum(v for k, v in domain_counts.items() if k != 'unknown')
    print(f"\nDomain extraction: {known_count}/{len(dataset)} episodes matched ({known_count/len(dataset)*100:.1f}%)")
    print(f"Domain distribution: {dict(domain_counts.most_common())}")

    # --- Analysis 2: Bigram novelty score ---
    global_bigrams = Counter()
    episode_bigrams = {}
    episode_action_types = {}

    for eid, ep in dataset.items():
        types = [s['action_content']['action'] for s in ep['steps']
                 if s['action_content']['action']]  # filter empty
        episode_action_types[eid] = types
        bgs = compute_action_bigrams(types)
        episode_bigrams[eid] = bgs
        for bg in bgs:
            global_bigrams[bg] += 1

    total_bigrams = sum(global_bigrams.values())
    print(f"\nTotal unique bigrams: {len(global_bigrams)}, total count: {total_bigrams}")

    # Novelty score = average log-inverse frequency of bigrams
    episode_novelty = {}
    for eid, bgs in episode_bigrams.items():
        if not bgs:
            episode_novelty[eid] = 0.0
            continue
        inv_freqs = []
        for bg in bgs:
            freq = global_bigrams[bg] / total_bigrams
            inv_freqs.append(-math.log(freq + 1e-10))
        episode_novelty[eid] = float(np.mean(inv_freqs))

    novelty_values = list(episode_novelty.values())
    print(f"Novelty score: mean={np.mean(novelty_values):.3f}, std={np.std(novelty_values):.3f}, "
          f"min={np.min(novelty_values):.3f}, max={np.max(novelty_values):.3f}")

    # --- Analysis 3: Novelty-binned accuracy ---
    # Only use episodes that appear in eval results
    eval_eids_with_novelty = [e for e in baseline_eps if e in episode_novelty]
    sorted_eids = sorted(eval_eids_with_novelty, key=lambda x: episode_novelty[x])
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
        bin_sft_acc = []
        bin_baseline_type_acc = []
        bin_sft_type_acc = []

        for eid in bin_eids:
            if eid in baseline_eps:
                bin_baseline_acc.append(compute_step_accuracy(baseline_eps[eid]))
                bin_baseline_type_acc.append(compute_type_accuracy(baseline_eps[eid]))
            if eid in sft_eps:
                bin_sft_acc.append(compute_step_accuracy(sft_eps[eid]))
                bin_sft_type_acc.append(compute_type_accuracy(sft_eps[eid]))

        bl_mean = float(np.mean(bin_baseline_acc)) if bin_baseline_acc else 0
        sf_mean = float(np.mean(bin_sft_acc)) if bin_sft_acc else 0
        bl_type_mean = float(np.mean(bin_baseline_type_acc)) if bin_baseline_type_acc else 0
        sf_type_mean = float(np.mean(bin_sft_type_acc)) if bin_sft_type_acc else 0

        novelty_bins[bin_label] = {
            'n_episodes': len(bin_eids),
            'novelty_range': [float(np.min(bin_novelty)), float(np.max(bin_novelty))],
            'novelty_mean': float(np.mean(bin_novelty)),
            'baseline_step_acc': bl_mean,
            'sft_step_acc': sf_mean,
            'gap': sf_mean - bl_mean,
            'baseline_type_acc': bl_type_mean,
            'sft_type_acc': sf_type_mean,
            'type_gap': sf_type_mean - bl_type_mean,
        }

    print("\nNovelty-binned accuracy:")
    for q, d in novelty_bins.items():
        print(f"  {q}: novelty={d['novelty_mean']:.2f}, bl_acc={d['baseline_step_acc']:.3f}, "
              f"sft_acc={d['sft_step_acc']:.3f}, gap={d['gap']:.3f}, n={d['n_episodes']}")

    # --- Analysis 4: Difficulty-controlled analysis ---
    # Within same SFT-accuracy bins, further split by novelty
    difficulty_controlled = {}
    sft_type_accs = {}
    for eid in eval_eids_with_novelty:
        if eid in sft_eps:
            sft_type_accs[eid] = compute_type_accuracy(sft_eps[eid])

    sorted_by_sft = sorted(sft_type_accs.keys(), key=lambda x: sft_type_accs[x])
    tertile_size = len(sorted_by_sft) // 3

    for t in range(3):
        start = t * tertile_size
        end = (t + 1) * tertile_size if t < 2 else len(sorted_by_sft)
        tertile_eids = sorted_by_sft[start:end]
        tertile_label = ['low_sft', 'mid_sft', 'high_sft'][t]

        tertile_novelties = [(e, episode_novelty.get(e, 0)) for e in tertile_eids]
        median_novelty = float(np.median([n for _, n in tertile_novelties]))

        low_novelty = [e for e, n in tertile_novelties if n <= median_novelty]
        high_novelty = [e for e, n in tertile_novelties if n > median_novelty]

        def compute_group_stats(eids):
            bl_accs = [compute_step_accuracy(baseline_eps[e]) for e in eids if e in baseline_eps]
            sf_accs = [compute_step_accuracy(sft_eps[e]) for e in eids if e in sft_eps]
            bl_type = [compute_type_accuracy(baseline_eps[e]) for e in eids if e in baseline_eps]
            sf_type = [compute_type_accuracy(sft_eps[e]) for e in eids if e in sft_eps]
            return {
                'n': len(eids),
                'baseline_step_acc': float(np.mean(bl_accs)) if bl_accs else 0,
                'sft_step_acc': float(np.mean(sf_accs)) if sf_accs else 0,
                'baseline_type_acc': float(np.mean(bl_type)) if bl_type else 0,
                'sft_type_acc': float(np.mean(sf_type)) if sf_type else 0,
                'gap': (float(np.mean(sf_accs)) - float(np.mean(bl_accs))) if bl_accs and sf_accs else 0,
            }

        difficulty_controlled[tertile_label] = {
            'sft_acc_range': [float(sft_type_accs[tertile_eids[0]]),
                              float(sft_type_accs[tertile_eids[-1]])],
            'low_novelty': compute_group_stats(low_novelty),
            'high_novelty': compute_group_stats(high_novelty),
            'novelty_median': median_novelty,
        }

    print("\nDifficulty-controlled analysis:")
    for t, d in difficulty_controlled.items():
        lo = d['low_novelty']
        hi = d['high_novelty']
        print(f"  {t} (sft range {d['sft_acc_range'][0]:.2f}-{d['sft_acc_range'][1]:.2f}):")
        print(f"    Low novelty:  bl={lo['baseline_step_acc']:.3f}, sft={lo['sft_step_acc']:.3f}, gap={lo['gap']:.3f} (n={lo['n']})")
        print(f"    High novelty: bl={hi['baseline_step_acc']:.3f}, sft={hi['sft_step_acc']:.3f}, gap={hi['gap']:.3f} (n={hi['n']})")

    # --- Analysis 5: Domain-frequency analysis ---
    domain_stats = {}
    for domain in ['excel', 'word', 'ppt']:
        domain_eids = [e for e in eval_eids_with_novelty if eval_domains.get(e) == domain]
        if len(domain_eids) < 3:
            continue
        bl_accs = [compute_step_accuracy(baseline_eps[e]) for e in domain_eids if e in baseline_eps]
        sf_accs = [compute_step_accuracy(sft_eps[e]) for e in domain_eids if e in sft_eps]
        bl_type = [compute_type_accuracy(baseline_eps[e]) for e in domain_eids if e in baseline_eps]
        sf_type = [compute_type_accuracy(sft_eps[e]) for e in domain_eids if e in sft_eps]
        domain_stats[domain] = {
            'count': len(domain_eids),
            'dataset_count': domain_counts.get(domain, 0),
            'baseline_step_acc': float(np.mean(bl_accs)) if bl_accs else 0,
            'sft_step_acc': float(np.mean(sf_accs)) if sf_accs else 0,
            'baseline_type_acc': float(np.mean(bl_type)) if bl_type else 0,
            'sft_type_acc': float(np.mean(sf_type)) if sf_type else 0,
            'gap': (float(np.mean(sf_accs)) - float(np.mean(bl_accs))) if bl_accs and sf_accs else 0,
        }

    # Correlation between domain frequency and baseline accuracy
    if len(domain_stats) >= 2:
        freqs = [d['dataset_count'] for d in domain_stats.values()]
        accs = [d['baseline_step_acc'] for d in domain_stats.values()]
        if len(freqs) > 2:
            corr = float(np.corrcoef(np.log(np.array(freqs) + 1), accs)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    print(f"\nDomain-frequency correlation (log_freq vs baseline_acc): r={corr:.3f}")
    for domain, d in domain_stats.items():
        print(f"  {domain}: n={d['count']}, bl={d['baseline_step_acc']:.3f}, sft={d['sft_step_acc']:.3f}, gap={d['gap']:.3f}")

    # --- Analysis 6: Action type diversity ---
    diversity_stats = defaultdict(lambda: {'baseline_accs': [], 'sft_accs': [], 'eids': []})
    for eid in eval_eids_with_novelty:
        types = episode_action_types.get(eid, [])
        n_unique = len(set(types))
        diversity_stats[n_unique]['eids'].append(eid)
        if eid in baseline_eps:
            diversity_stats[n_unique]['baseline_accs'].append(compute_step_accuracy(baseline_eps[eid]))
        if eid in sft_eps:
            diversity_stats[n_unique]['sft_accs'].append(compute_step_accuracy(sft_eps[eid]))

    diversity_results = {}
    for n_unique in sorted(diversity_stats.keys()):
        d = diversity_stats[n_unique]
        bl = d['baseline_accs']
        sf = d['sft_accs']
        diversity_results[str(n_unique)] = {
            'n_episodes': len(d['eids']),
            'baseline_step_acc': float(np.mean(bl)) if bl else 0,
            'sft_step_acc': float(np.mean(sf)) if sf else 0,
            'gap': (float(np.mean(sf)) - float(np.mean(bl))) if bl and sf else 0,
        }

    print("\nAction type diversity:")
    for nu, d in diversity_results.items():
        print(f"  {nu} unique types: bl={d['baseline_step_acc']:.3f}, sft={d['sft_step_acc']:.3f}, gap={d['gap']:.3f} (n={d['n_episodes']})")

    # Assemble output
    output = {
        'domain_extraction': {
            'coverage': known_count / len(dataset) if dataset else 0,
            'n_matched': known_count,
            'n_total': len(dataset),
            'domain_distribution': dict(domain_counts.most_common()),
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
        'domain_frequency': {
            'correlation_log_freq_vs_acc': corr,
            'per_domain': domain_stats,
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
            'domain_freq_correlation': corr,
        },
    }

    save_json(output, os.path.join(args.output_dir, 'q5_results.json'))
    print(f"\nResults saved to {os.path.join(args.output_dir, 'q5_results.json')}")


if __name__ == "__main__":
    main()
