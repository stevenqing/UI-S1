"""
Convert GUI-360 evaluator's native JSON format to the unified JSONL format
expected by step2_classify_errors.py and step1_analyze_cascade.py.
"""
import json
import os
import sys


def convert(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    trajectories = data['trajectory_results']
    print(f"Loaded {len(trajectories)} trajectories from {input_path}")

    with open(output_path, 'w') as fout:
        for t in trajectories:
            steps = []
            for s in t.get('step_results', []):
                steps.append({
                    'step_id': s['step_num'] - 1,  # 0-indexed
                    'success': s['success'],
                    'type_match': s['function_match'],
                    'grounding_match': s['args_match'],
                    'pred_action': {
                        'action': s.get('predicted_function', ''),
                        'args': s.get('predicted_args', {}),
                        'status': s.get('predicted_status', ''),
                    },
                    'gt_action_type': s.get('ground_truth_function', 'unknown'),
                    'step_instruction': s.get('thoughts', ''),
                })

            result = {
                'dataset': 'gui360',
                'trajectory_id': t['trajectory_id'],
                'goal': t.get('trajectory_id', ''),  # GUI-360 doesn't store goal text in results
                'num_steps': t['num_steps'],
                'num_evaluated': len(steps),
                'mode': 'natural_cascade',
                'steps': steps,
                'trajectory_success': t['trajectory_success'],
                'first_error_step': t.get('first_error_step'),
                'progress_rate': t['progress_rate'],
                'scattered_progress_rate': t['scattered_progress_rate'],
                'domain': t.get('domain', ''),
                'category': t.get('category', ''),
            }
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Wrote {len(trajectories)} trajectories to {output_path}")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/gui360/nostop_20260319_170859/ar_evaluation_results_20260319_182012.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/gui360/gui360_nostop_results.jsonl"
    convert(input_path, output_path)
