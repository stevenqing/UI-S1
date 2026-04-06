"""Eval U10: History-Aware Verifier Multi-Agent AR Trajectory Evaluation.

Key improvement over U7 (stateless verifier):
  U10 uses a 2D policy: (agreement × prev_step_correct) to decide PASS/FAIL.

Architecture per step:
1. Generate K=3 quick samples → compute agreement.
2. Look up prev_step_correct from trajectory history.
3. Apply 2D policy:
   - High-agree + prev_correct → PASS (use greedy, ~79% accuracy)
   - High-agree + prev_wrong → CONDITIONAL PASS (use greedy but cautious)
   - Med-agree + prev_correct → PASS (use greedy, ~52%)
   - Med-agree + prev_wrong → FAIL → resample K=5, majority vote (~39.5% → ~49.5%)
   - Low-agree → FAIL → resample K=5 (history doesn't help at low agreement)
4. Continue AR (stop on first extract_match=False).

Pre-registered prediction (from Markov model): TSR ~16.61% (+0.54pp over baseline).
"""

import argparse
import copy
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from ac_utils import (
    load_ac_trajectories, fix_line, init_format, save_jsonl, save_json,
    compute_trajectory_metrics, length_bucket,
    evaluate_android_control_action,
    find_last_image_ele, slim_messages, safe_parse_response, _json_default,
    call_mobile_agent_vllm, generate_k_samples_fast,
)

result_lock = Lock()
fm = None


def get_agreement_bin(agreement):
    """Map agreement value to bin name."""
    if agreement < 0.5:
        return 'low'
    elif agreement < 0.7:
        return 'med'
    elif agreement < 0.9:
        return 'high'
    else:
        return 'vhigh'


def compute_agreement(samples):
    """Compute action type agreement from K samples."""
    action_types = []
    for s in samples:
        if s['pred_action'] and s['parse_ok']:
            action_types.append(s['pred_action'].get('action', 'unknown'))

    if not action_types:
        return 0.0, 'unknown', {}

    type_counter = Counter(action_types)
    voted_type = type_counter.most_common(1)[0][0]
    agreement = type_counter.most_common(1)[0][1] / len(action_types)

    return agreement, voted_type, dict(type_counter)


def history_aware_policy(agreement, prev_correct, step_num):
    """2D verification policy based on agreement and error history.

    Returns:
        decision: 'PASS', 'CONDITIONAL_PASS', or 'FAIL'
    """
    abin = get_agreement_bin(agreement)

    # Step 0: no history available, use agreement-only policy
    if step_num == 0:
        if abin in ['vhigh', 'high']:
            return 'PASS'
        else:
            return 'FAIL'

    # Steps 1+: 2D policy
    if abin == 'vhigh':
        if prev_correct:
            return 'PASS'          # 79% accuracy, reliable
        else:
            return 'CONDITIONAL_PASS'  # 72%, still decent but cautious

    elif abin == 'high':
        if prev_correct:
            return 'PASS'          # 63%, reasonable
        else:
            return 'CONDITIONAL_PASS'  # 55%, worth checking

    elif abin == 'med':
        if prev_correct:
            return 'PASS'          # 52%, borderline but history says OK
        else:
            return 'FAIL'          # 39.5% → HIGHEST ROI intervention point (+12.5pp gap)

    else:  # low
        return 'FAIL'              # 38-39%, history doesn't help here


def majority_vote_select(samples):
    """Select action by majority vote on action type, then pick first match."""
    action_types = []
    for s in samples:
        if s['pred_action'] and s['parse_ok']:
            action_types.append(s['pred_action'].get('action', 'unknown'))

    if not action_types:
        return samples[0] if samples else None, 'all_failed', {}

    type_counter = Counter(action_types)
    voted_type = type_counter.most_common(1)[0][0]
    agreement = type_counter.most_common(1)[0][1] / len(action_types)

    for s in samples:
        if s['pred_action'] and s['pred_action'].get('action') == voted_type:
            return s, voted_type, {
                'agreement': agreement,
                'type_counts': dict(type_counter),
                'voted_type': voted_type,
            }

    return samples[0], voted_type, {'agreement': agreement}


def process_episode(episode, args):
    """Process a single episode with History-Aware Verifier AR evaluation."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    goal = episode['goal']
    state = None
    model_response = None
    step_results = []
    prev_correct = None  # Track error history

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']

            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = slim_messages(
                messages=state['messages'],
                num_image_limit=args.n_history_image_limit,
            )

            image_ele_result = find_last_image_ele(messages)
            width = image_ele_result[1]
            height = image_ele_result[2]
            resized_width = image_ele_result[3]
            resized_height = image_ele_result[4]

            # --- Step 1: Generate K_probe quick samples for agreement ---
            probe_samples = generate_k_samples_fast(
                messages, args.model_name, args.K_probe, args.probe_temperature, fm,
            )
            agreement, probe_voted_type, type_counts = compute_agreement(probe_samples)

            # --- Step 2: Apply 2D history-aware policy ---
            decision = history_aware_policy(agreement, prev_correct, step_id)

            resampled = False
            pred_action = None
            model_response = None

            if decision == 'PASS':
                # Use first probe sample (greedy-like) as the action
                if probe_samples and probe_samples[0]['pred_action'] and probe_samples[0]['parse_ok']:
                    pred_action = probe_samples[0]['pred_action']
                    model_response = probe_samples[0]['response']
                else:
                    # Fallback: call greedy
                    actor_response = call_mobile_agent_vllm(
                        messages=messages, model_name=args.model_name,
                    )
                    actor_pred = safe_parse_response(fm, actor_response)
                    pred_action = actor_pred['action_content']
                    model_response = actor_response

            elif decision == 'CONDITIONAL_PASS':
                # Use first probe sample but could enhance later
                if probe_samples and probe_samples[0]['pred_action'] and probe_samples[0]['parse_ok']:
                    pred_action = probe_samples[0]['pred_action']
                    model_response = probe_samples[0]['response']
                else:
                    actor_response = call_mobile_agent_vllm(
                        messages=messages, model_name=args.model_name,
                    )
                    actor_pred = safe_parse_response(fm, actor_response)
                    pred_action = actor_pred['action_content']
                    model_response = actor_response

            elif decision == 'FAIL':
                resampled = True
                # Generate additional K_resample samples for majority vote
                resample_samples = generate_k_samples_fast(
                    messages, args.model_name, args.K_resample, args.resample_temperature, fm,
                )
                # Combine probe + resample samples for better majority vote
                all_samples = probe_samples + resample_samples
                selected, voted_type, vote_info = majority_vote_select(all_samples)
                agreement = vote_info.get('agreement', agreement)

                if selected and selected['pred_action']:
                    pred_action = selected['pred_action']
                    model_response = selected['response']
                else:
                    # Fallback
                    if probe_samples and probe_samples[0]['pred_action']:
                        pred_action = probe_samples[0]['pred_action']
                        model_response = probe_samples[0]['response']

            # Fallback if still no action
            if pred_action is None:
                actor_response = call_mobile_agent_vllm(
                    messages=messages, model_name=args.model_name,
                )
                actor_pred = safe_parse_response(fm, actor_response)
                pred_action = actor_pred['action_content']
                model_response = actor_response

            # --- Evaluate ---
            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height,
            )

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'decision': decision,
                'resampled': resampled,
                'agreement': agreement,
                'agreement_bin': get_agreement_bin(agreement),
                'prev_correct': prev_correct,
                'type_counts': type_counts,
            })

            # Update history
            prev_correct = extract_match

            # Stop on first failure
            if not extract_match:
                break

    except Exception as e:
        print(f"Error episode {episode.get('episode_id', '?')}: {e}")

    correct_steps = sum(1 for s in step_results if s['extract_match'])
    task_success = (correct_steps == num_steps and len(step_results) == num_steps)

    result = {
        'episode_id': episode.get('episode_id'),
        'goal': goal,
        'num_steps': num_steps,
        'task_success': task_success,
        'final_step_id': correct_steps,
        'step_results': step_results,
        'length_bucket': length_bucket(num_steps),
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'history_verifier_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'history_verifier_results.jsonl')

    # Resume support: load already-completed episode IDs
    completed_ids = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    ep = json.loads(line)
                    completed_ids.add(ep.get('episode_id'))
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(completed_ids)} episodes already completed")
    elif os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    # Filter out already completed episodes
    if completed_ids:
        data = [ep for ep in data if ep.get('episode_id') not in completed_ids]
    print(f"Loaded {len(data)} remaining episodes. Running U10 History-Aware Verifier AR "
          f"(K_probe={args.K_probe}, K_resample={args.K_resample})...")

    results = []
    total_episodes = len(data) + len(completed_ids)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                done = len(results) + len(completed_ids)
                if len(results) % 50 == 0:
                    metrics = compute_trajectory_metrics(results)
                    n_pass = sum(1 for r in results for s in r['step_results'] if s['decision'] == 'PASS')
                    n_fail = sum(1 for r in results for s in r['step_results'] if s['decision'] == 'FAIL')
                    n_cond = sum(1 for r in results for s in r['step_results'] if s['decision'] == 'CONDITIONAL_PASS')
                    total = n_pass + n_fail + n_cond
                    print(f"Progress: {done}/{total_episodes} | TSR: {metrics['tsr']:.3f} | "
                          f"PASS:{n_pass}/{total} COND:{n_cond}/{total} FAIL:{n_fail}/{total}")
            except Exception as e:
                print(f"Exception: {e}")

    # Reload all results (including previously completed) for final summary
    all_results = []
    with open(out_path) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    results = all_results

    metrics = compute_trajectory_metrics(results)

    # Decision stats
    total_steps = sum(len(r['step_results']) for r in results)
    decision_counts = {'PASS': 0, 'CONDITIONAL_PASS': 0, 'FAIL': 0}
    for r in results:
        for s in r['step_results']:
            decision_counts[s['decision']] = decision_counts.get(s['decision'], 0) + 1

    # Agreement bin distribution
    bin_counts = {}
    for r in results:
        for s in r['step_results']:
            abin = s.get('agreement_bin', 'unknown')
            bin_counts[abin] = bin_counts.get(abin, 0) + 1

    # History effect: accuracy when prev_correct=True vs False
    hist_stats = {'prev_true': {'correct': 0, 'total': 0}, 'prev_false': {'correct': 0, 'total': 0}}
    for r in results:
        for s in r['step_results']:
            if s['prev_correct'] is True:
                hist_stats['prev_true']['total'] += 1
                if s['extract_match']:
                    hist_stats['prev_true']['correct'] += 1
            elif s['prev_correct'] is False:
                hist_stats['prev_false']['total'] += 1
                if s['extract_match']:
                    hist_stats['prev_false']['correct'] += 1

    summary = {
        'model': args.model_name,
        'experiment': 'U10_history_aware_verifier_AR',
        'K_probe': args.K_probe,
        'K_resample': args.K_resample,
        'probe_temperature': args.probe_temperature,
        'resample_temperature': args.resample_temperature,
        'total_episodes': len(results),
        **metrics,
        'total_steps_evaluated': total_steps,
        'decision_counts': decision_counts,
        'decision_rates': {k: v / total_steps for k, v in decision_counts.items()} if total_steps > 0 else {},
        'agreement_bin_counts': bin_counts,
        'history_effect': {
            'prev_true_accuracy': hist_stats['prev_true']['correct'] / hist_stats['prev_true']['total']
            if hist_stats['prev_true']['total'] > 0 else 0,
            'prev_false_accuracy': hist_stats['prev_false']['correct'] / hist_stats['prev_false']['total']
            if hist_stats['prev_false']['total'] > 0 else 0,
            'prev_true_n': hist_stats['prev_true']['total'],
            'prev_false_n': hist_stats['prev_false']['total'],
        },
        'pre_registered_prediction': {
            'predicted_tsr': 0.1661,
            'predicted_delta_vs_baseline': 0.0054,
        },
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nU10 History-Aware Verifier AR completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Decisions: PASS={decision_counts['PASS']} COND={decision_counts['CONDITIONAL_PASS']} FAIL={decision_counts['FAIL']}")
    print(f"Agreement bins: {bin_counts}")
    print(f"\nPre-registered prediction: TSR=16.61%, delta=+0.54pp")
    print(f"Actual TSR: {metrics['tsr']*100:.2f}%")
    print(f"Prediction error: {abs(metrics['tsr'] - 0.1661)*100:.2f}pp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval U10: History-Aware Verifier Multi-Agent AR")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_u10_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--K_probe", type=int, default=3, help="Probe samples for agreement")
    parser.add_argument("--K_resample", type=int, default=5, help="Resample candidates on FAIL")
    parser.add_argument("--probe_temperature", type=float, default=0.6, help="Probe temperature")
    parser.add_argument("--resample_temperature", type=float, default=0.6, help="Resample temperature")
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()
    main(args)
