"""Shared utilities for adaptive replanning evaluation experiments (E1-E4).

Provides:
  - Unified Trajectory/Step dataclasses for AC and GUI-360
  - DatasetAdapter abstraction (ACAdapter, GUI360Adapter)
  - Boundary detection: oracle, fixed interval, agreement-based
  - Planner communication protocols (none / NL / structured / type_only / structured+progress)
  - Shared evaluation loop: run_replan_trajectory()
  - Replanning-specific metrics: compute_replan_metrics()
"""

import copy
import json
import os
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock

# ── Path setup ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'evaluation'))

GUI360_EVAL_DIR = os.path.join(PROJECT_ROOT, "train_GUI_360", "GUI-360-eval")

# ── Dataclasses ──

@dataclass
class Step:
    step_id: int
    gt_action_type: str
    gt_instruction: str
    screenshot_path: str
    raw_step: dict


@dataclass
class Trajectory:
    trajectory_id: str
    goal: str
    num_steps: int
    steps: list          # list[Step]
    raw_data: dict       # original format for dataset-specific ops
    dataset: str         # 'ac' or 'gui360'


# ── Action text formatting ──

def format_action_text(pred_action):
    """Create brief text description of an action (AC format)."""
    if pred_action is None:
        return "unknown action"
    action_type = pred_action.get('action', 'unknown')
    if action_type in ('click', 'long_press'):
        coord = pred_action.get('coordinate', [])
        return f"{action_type} at {coord}"
    elif action_type == 'type':
        text = pred_action.get('text', '')
        return f'type "{text}"'
    elif action_type == 'swipe':
        coord = pred_action.get('coordinate', [])
        direction = pred_action.get('direction', '')
        return f"swipe {direction} at {coord}"
    elif action_type == 'open':
        app = pred_action.get('app', '')
        return f'open "{app}"'
    elif action_type == 'wait':
        return "wait"
    elif action_type == 'system_button':
        button = pred_action.get('button', '')
        return f'press system button "{button}"'
    else:
        return f"{action_type}({json.dumps(pred_action, ensure_ascii=False)})"


def format_gui360_action_text(pred_function, pred_args):
    """Format GUI-360 prediction as text."""
    if not pred_function:
        return "unknown action"
    args_str = ", ".join(f"{k}={v}" for k, v in (pred_args or {}).items())
    return f"{pred_function}({args_str})"


# ── Instruction parsing (reused from eval_oracle_ablation.py) ──

ACTION_VERB_PREFIXES = [
    'click on the ', 'click on ', 'click the ', 'click ',
    'tap on the ', 'tap on ', 'tap the ', 'tap ',
    'long press on the ', 'long press on ', 'long press the ', 'long press ',
    'press and hold the ', 'press and hold on ', 'press and hold ',
    'swipe ', 'scroll ',
    'type in ', 'type ', 'enter the ', 'enter ', 'input the ', 'input ',
    'search for ', 'search ',
    'open the ', 'open ', 'launch the ', 'launch ',
    'press the ', 'press ',
    'go back to ', 'go back', 'navigate back',
    'wait for ', 'wait ',
]


def extract_target(step_instruction):
    """Extract target noun phrase from step instruction by stripping the action verb."""
    if not step_instruction:
        return ''
    instr = step_instruction.strip()
    instr_lower = instr.lower()
    for prefix in ACTION_VERB_PREFIXES:
        if instr_lower.startswith(prefix):
            target = instr[len(prefix):].strip()
            if target.endswith('.'):
                target = target[:-1].strip()
            return target if target else instr
    return instr


# ── Planner Communication Protocols ──

def generate_planner_message(trajectory, step_id, protocol, completed_summary=''):
    """Generate planner message at a boundary.

    Args:
        trajectory: Trajectory object
        step_id: current step index (boundary step)
        protocol: 'none' | 'nl_instruction' | 'structured' | 'type_only' | 'structured_progress'
        completed_summary: text summary of completed actions so far

    Returns:
        planner message string
    """
    step = trajectory.steps[step_id]
    gt_type = step.gt_action_type
    gt_instruction = step.gt_instruction
    goal = trajectory.goal

    if protocol == 'none':
        return goal

    if protocol == 'nl_instruction':
        return gt_instruction if gt_instruction else goal

    target = extract_target(gt_instruction)

    if protocol == 'structured':
        if target:
            return f"{gt_type}: {target}"
        return f"Perform a {gt_type} action."

    if protocol == 'type_only':
        return f"Perform a {gt_type} action."

    if protocol == 'structured_progress':
        base = f"{gt_type}: {target}" if target else f"Perform a {gt_type} action."
        remaining = trajectory.num_steps - step_id
        parts = [base]
        if completed_summary:
            parts.append(f"Completed so far: {completed_summary}")
        parts.append(f"Remaining steps: approximately {remaining}")
        return " | ".join(parts)

    raise ValueError(f"Unknown protocol: {protocol}")


# ── Boundary Detection ──

def detect_oracle_boundaries(trajectory):
    """Return step IDs where GT action type changes (always includes 0)."""
    boundaries = [0]
    for i in range(1, trajectory.num_steps):
        if trajectory.steps[i].gt_action_type != trajectory.steps[i - 1].gt_action_type:
            boundaries.append(i)
    return boundaries


def detect_fixed_interval_boundaries(trajectory, interval):
    """Return [0, K, 2K, ...] up to num_steps."""
    return list(range(0, trajectory.num_steps, interval))


def detect_agreement_boundary_ac(trajectory, step_id, args, threshold, K=5):
    """Check if agreement rate < threshold at this step (AC dataset).

    Returns True if this should be a boundary (low agreement = high uncertainty).
    """
    from ac_utils import (
        init_format, generate_k_samples_fast, fix_line,
        find_last_image_ele, slim_messages,
    )
    from x.data.agent.json import MOBILE_USE, OUTPUT_FORMAT, generate_prompt
    from x.qwen.image import make_qwen_image_item

    fm = init_format()
    fixed = trajectory.raw_data

    line_can_thought = fm.can_thought(fixed)
    _format = 'thought_action' if line_can_thought else 'only_action'
    system_prompt = MOBILE_USE.format(OUTPUT_FORMAT[_format], generate_prompt(fm.space))

    step = fixed['steps'][step_id]
    messages = [
        {'role': 'system', 'content': [{'text': system_prompt}]},
        {'role': 'user', 'content': [
            {'text': f"Overall Task: {fixed['goal']}\n\nPlease perform the next action."},
            make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None)),
        ]}
    ]
    messages = slim_messages(messages=messages, num_image_limit=2)

    samples = generate_k_samples_fast(messages, args.model_name, K, temperature=1.0, fm_obj=fm)

    action_types = [
        s['pred_action'].get('action', '?')
        for s in samples if s['pred_action'] and s['parse_ok']
    ]
    if not action_types:
        return True  # Can't parse -> high uncertainty -> boundary

    counter = Counter(action_types)
    agreement = counter.most_common(1)[0][1] / len(action_types)
    return agreement < threshold


def detect_agreement_boundary_gui360(trajectory, step_id, adapter, args, threshold, K=5):
    """Check if agreement rate < threshold at this step (GUI-360 dataset).

    Returns True if this should be a boundary.
    """
    step = trajectory.steps[step_id]
    sample = step.raw_step
    goal = trajectory.goal

    predictions = []

    def _predict_one(_):
        try:
            pf, _, _ = adapter._predict_with_goal_raw(sample, goal, temperature=1.0)
            return pf
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=K) as executor:
        futures = [executor.submit(_predict_one, i) for i in range(K)]
        for f in as_completed(futures):
            pred_func = f.result()
            if pred_func:
                predictions.append(pred_func)

    if not predictions:
        return True

    counter = Counter(predictions)
    agreement = counter.most_common(1)[0][1] / len(predictions)
    return agreement < threshold


# ── Dataset Adapters ──

class DatasetAdapter(ABC):
    @abstractmethod
    def load_trajectories(self, args):
        """Load and return list of Trajectory objects."""
        pass

    @abstractmethod
    def predict_step(self, trajectory, step_id, instruction, action_history, args):
        """Predict action for a step.

        Args:
            trajectory: Trajectory object
            step_id: current step index
            instruction: text instruction (planner message or goal)
            action_history: list of action text strings since last boundary
            args: argparse namespace

        Returns:
            dict with keys: pred, dims, action_text, response, thought
        """
        pass

    @abstractmethod
    def evaluate_step(self, pred, trajectory, step_id, dims):
        """Evaluate prediction. Returns (type_match: bool, extract_match: bool)."""
        pass

    def detect_agreement_boundary(self, trajectory, step_id, args, threshold, K=5):
        """Check if this step should be a boundary based on agreement."""
        raise NotImplementedError


class ACAdapter(DatasetAdapter):
    def __init__(self):
        from ac_utils import init_format
        self.fm = init_format()

    def load_trajectories(self, args):
        from ac_utils import load_ac_trajectories, fix_line
        episodes = load_ac_trajectories(
            jsonl_path=args.jsonl_file,
            max_episodes=getattr(args, 'max_episodes', None),
        )
        trajectories = []
        for ep in episodes:
            fixed = fix_line(copy.deepcopy(ep))
            steps = []
            for i, s in enumerate(fixed['steps']):
                steps.append(Step(
                    step_id=i,
                    gt_action_type=s['action_content'].get('action', 'unknown'),
                    gt_instruction=s.get('step_instruction', ''),
                    screenshot_path=s['screenshot'],
                    raw_step=s,
                ))
            trajectories.append(Trajectory(
                trajectory_id=ep.get('episode_id', str(len(trajectories))),
                goal=ep['goal'],
                num_steps=len(steps),
                steps=steps,
                raw_data=fixed,
                dataset='ac',
            ))
        return trajectories

    def predict_step(self, trajectory, step_id, instruction, action_history, args):
        from ac_utils import (
            call_mobile_agent_vllm, safe_parse_response,
            find_last_image_ele, slim_messages,
        )
        from x.data.agent.json import MOBILE_USE, OUTPUT_FORMAT, generate_prompt
        from x.qwen.image import make_qwen_image_item

        fixed = trajectory.raw_data
        step = fixed['steps'][step_id]

        line_can_thought = self.fm.can_thought(fixed)
        _format = 'thought_action' if line_can_thought else 'only_action'
        system_prompt = MOBILE_USE.format(OUTPUT_FORMAT[_format], generate_prompt(self.fm.space))

        messages = [{'role': 'system', 'content': [{'text': system_prompt}]}]

        user_content = []
        text_parts = []
        text_parts.append(f"Overall Task: {trajectory.goal}")

        if instruction and instruction != trajectory.goal:
            text_parts.append(f"\nCurrent Step Instruction: {instruction}")

        if action_history:
            text_parts.append(f"\nCompleted actions ({len(action_history)} step(s)):")
            for i, action_text in enumerate(action_history):
                text_parts.append(f"  Step {i + 1}: {action_text}")
            text_parts.append(f"\nPlease perform step {len(action_history) + 1}.")
        else:
            text_parts.append(f"\nThis is the first step. Please begin the task.")

        format_instruct = f"Output Format: {OUTPUT_FORMAT[_format]}"
        text_parts.append(f"\n{format_instruct}")
        user_content.append({'text': '\n'.join(text_parts)})

        if step_id == 0:
            user_content.append({
                'text': "If the query asks a question, please answer the question through the answer action before terminating the process.\n"
            })

        image_ele = make_qwen_image_item(
            step['screenshot'], image=step.get('screenshot_pil', None)
        )
        user_content.append(image_ele)
        messages.append({'role': 'user', 'content': user_content})

        messages = slim_messages(messages=messages, num_image_limit=2)
        _, width, height, resized_width, resized_height = find_last_image_ele(messages)

        model_response = call_mobile_agent_vllm(
            messages=messages, model_name=args.model_name
        )

        pred = safe_parse_response(self.fm, model_response)
        pred_action = pred['action_content']
        thought_text = pred.get('think', '')
        action_text = format_action_text(pred_action)

        return {
            'pred': pred_action,
            'dims': (width, height, resized_width, resized_height),
            'action_text': action_text,
            'response': model_response,
            'thought': thought_text,
        }

    def evaluate_step(self, pred, trajectory, step_id, dims):
        from ac_utils import evaluate_android_control_action
        check_options = trajectory.raw_data['steps'][step_id]['check_options']
        w, h, rw, rh = dims
        type_match, extract_match = evaluate_android_control_action(
            pred, check_options, w, h, rw, rh
        )
        return type_match, extract_match

    def detect_agreement_boundary(self, trajectory, step_id, args, threshold, K=5):
        return detect_agreement_boundary_ac(trajectory, step_id, args, threshold, K)


class GUI360Adapter(DatasetAdapter):
    def __init__(self):
        self._model = None
        sys.path.insert(0, GUI360_EVAL_DIR)

    def _ensure_model(self, args):
        if self._model is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "qwen2_5_vl_7b",
                os.path.join(GUI360_EVAL_DIR, "models", "qwen2.5_vl_7b.py"),
            )
            qwen_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qwen_module)
            self._model = qwen_module.Qwen25VL7B(
                api_url=args.api_url, model_name=args.model_name,
                coordinate_system="absolute", resize_factor=28,
            )

    def load_trajectories(self, args):
        from eval_gui360_todo2_subtask import load_gui360_samples_with_subtask
        samples = load_gui360_samples_with_subtask(
            getattr(args, 'root_dir', os.path.join(PROJECT_ROOT, 'datasets/GUI-360/test')),
            max_samples=getattr(args, 'max_samples', None),
        )

        # Group by trajectory_id and sort by step_index
        traj_groups = defaultdict(list)
        for s in samples:
            traj_groups[s['trajectory_id']].append(s)

        trajectories = []
        for tid, group in sorted(traj_groups.items()):
            group.sort(key=lambda x: x['step_index'])
            steps = []
            for i, s in enumerate(group):
                steps.append(Step(
                    step_id=i,
                    gt_action_type=s.get('gt_function', 'unknown'),
                    gt_instruction=s.get('subtask', ''),
                    screenshot_path=s.get('screenshot_clean', ''),
                    raw_step=s,
                ))
            trajectories.append(Trajectory(
                trajectory_id=tid,
                goal=group[0]['request'],
                num_steps=len(steps),
                steps=steps,
                raw_data={'samples': group},
                dataset='gui360',
            ))
        return trajectories

    def _predict_with_goal_raw(self, sample, goal_text, temperature=0.0):
        """Raw prediction: returns (pred_function, pred_args, pred_status)."""
        system_prompt, user_prompt = self._model.construct_action_prompt(
            instruction=goal_text,
            history="\n".join(sample["previous_actions"]) if sample["previous_actions"] else "",
            actions=sample["actions_str"],
            resolution=None,
        )

        # Pre-read image with retry to avoid Lustre PermissionError
        try:
            base64_image = self._read_image_base64(sample["screenshot_clean"])
            image_url = f"data:image/png;base64,{base64_image}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            result = self._model.client.chat.completions.create(
                messages=messages,
                model=self._model.model_name,
                max_tokens=4096,
                temperature=temperature,
            )
            raw_response = result.choices[0].message.content
        except Exception as e:
            print(f"Error during GUI-360 prediction: {e}")
            raw_response = ""

        return self._model.parse_action(raw_response)

    def _read_image_base64(self, image_path, max_retries=3):
        """Read image and convert to base64, with retries for Lustre FS issues."""
        import base64
        from io import BytesIO
        from PIL import Image
        import time

        for attempt in range(max_retries):
            try:
                image = Image.open(image_path)
                processed = self._model._preprocess_image_with_smart_resize(image)
                buffer = BytesIO()
                processed.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise

    def predict_step(self, trajectory, step_id, instruction, action_history, args):
        self._ensure_model(args)

        sample = trajectory.steps[step_id].raw_step

        # Build history from action_history (overrides sample's previous_actions)
        history_text = "\n".join(
            f"Step {i + 1}: {a}" for i, a in enumerate(action_history)
        ) if action_history else ""

        system_prompt, user_prompt = self._model.construct_action_prompt(
            instruction=instruction,
            history=history_text,
            actions=sample["actions_str"],
            resolution=None,
        )

        # Pre-read image with retry to avoid Lustre PermissionError
        try:
            base64_image = self._read_image_base64(sample["screenshot_clean"])
            image_url = f"data:image/png;base64,{base64_image}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            result = self._model.client.chat.completions.create(
                messages=messages,
                model=self._model.model_name,
                max_tokens=4096,
                temperature=0.0,
            )
            raw_response = result.choices[0].message.content
        except Exception as e:
            print(f"Error during GUI-360 prediction: {e}")
            raw_response = ""

        pred_function, pred_args, pred_status = self._model.parse_action(raw_response)
        action_text = format_gui360_action_text(pred_function, pred_args)

        return {
            'pred': (pred_function, pred_args, pred_status),
            'dims': None,
            'action_text': action_text,
            'response': raw_response,
            'thought': '',
        }

    def evaluate_step(self, pred, trajectory, step_id, dims):
        from eval_gui360_todo2_subtask import evaluate_pred
        pred_function, pred_args, pred_status = pred
        sample = trajectory.steps[step_id].raw_step
        function_match, args_match = evaluate_pred(pred_function, pred_args, sample)
        return function_match, args_match

    def detect_agreement_boundary(self, trajectory, step_id, args, threshold, K=5):
        self._ensure_model(args)
        return detect_agreement_boundary_gui360(trajectory, step_id, self, args, threshold, K)


def get_adapter(dataset_name):
    """Factory: return appropriate DatasetAdapter."""
    if dataset_name == 'ac':
        return ACAdapter()
    elif dataset_name == 'gui360':
        return GUI360Adapter()
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Shared Evaluation Loop ──

def run_replan_trajectory(trajectory, adapter, args, boundaries, protocol):
    """Run one trajectory with given boundaries and protocol.

    At boundary steps: context resets, planner generates message.
    Between boundaries: action history accumulates since last boundary.
    All steps evaluated (teacher-forced, no break on failure).

    Args:
        trajectory: Trajectory object
        adapter: DatasetAdapter instance
        args: argparse namespace
        boundaries: list of boundary step IDs
        protocol: planner communication protocol name

    Returns:
        dict with trajectory_id, step_results, metrics
    """
    action_history = []
    step_results = []
    current_planner_msg = None
    completed_actions = []
    boundary_set = set(boundaries)

    for step_id in range(trajectory.num_steps):
        is_boundary = step_id in boundary_set

        if is_boundary:
            # Context reset
            action_history = []
            completed_summary = "; ".join(completed_actions[-5:]) if completed_actions else ""
            current_planner_msg = generate_planner_message(
                trajectory, step_id, protocol, completed_summary
            )

        instruction = current_planner_msg if current_planner_msg else trajectory.goal

        try:
            result = adapter.predict_step(
                trajectory, step_id, instruction, action_history, args
            )
            type_match, extract_match = adapter.evaluate_step(
                result['pred'], trajectory, step_id, result['dims']
            )
            action_history.append(result['action_text'])
            completed_actions.append(result['action_text'])

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'is_boundary': is_boundary,
                'action_text': result['action_text'],
                'gt_action_type': trajectory.steps[step_id].gt_action_type,
                'instruction_given': instruction[:200] if instruction else '',
            })
        except Exception as e:
            action_history.append('error')
            completed_actions.append('error')
            step_results.append({
                'step_num': step_id,
                'type_match': False,
                'extract_match': False,
                'is_boundary': is_boundary,
                'action_text': 'error',
                'gt_action_type': trajectory.steps[step_id].gt_action_type,
                'error': str(e),
            })

    total_correct = sum(1 for s in step_results if s['extract_match'])
    return {
        'trajectory_id': trajectory.trajectory_id,
        'goal': trajectory.goal,
        'num_steps': trajectory.num_steps,
        'step_results': step_results,
        'total_correct': total_correct,
        'step_accuracy': total_correct / len(step_results) if step_results else 0,
        'n_boundaries': len(boundaries),
    }


# ── Metrics ──

def length_bucket(n):
    """Assign trajectory length to bucket."""
    if n <= 3:
        return 'short(1-3)'
    elif n <= 7:
        return 'medium(4-7)'
    elif n <= 15:
        return 'long(8-15)'
    else:
        return 'vlong(16+)'


def compute_replan_metrics(results):
    """Compute replanning-specific metrics.

    Each result should have: trajectory_id, num_steps, step_results.
    Each step in step_results: step_num, type_match, extract_match, is_boundary.
    """

    if not results:
        return {}

    n = len(results)

    # Aggregate all step results
    all_step_results = []
    total_steps = 0
    total_correct = 0
    type_correct = 0
    for r in results:
        for s in r['step_results']:
            total_steps += 1
            total_correct += int(s['extract_match'])
            type_correct += int(s['type_match'])
            all_step_results.append(s)

    step_accuracy = total_correct / total_steps if total_steps > 0 else 0
    type_accuracy = type_correct / total_steps if total_steps > 0 else 0

    # TSR: all steps correct
    tsr_count = sum(
        1 for r in results
        if all(s['extract_match'] for s in r['step_results']) and len(r['step_results']) == r['num_steps']
    )
    tsr = tsr_count / n

    # Boundary vs within-phase accuracy
    boundary_correct = sum(1 for s in all_step_results if s.get('is_boundary') and s['extract_match'])
    boundary_total = sum(1 for s in all_step_results if s.get('is_boundary'))
    within_correct = sum(1 for s in all_step_results if not s.get('is_boundary') and s['extract_match'])
    within_total = sum(1 for s in all_step_results if not s.get('is_boundary'))

    # Planner call count
    n_planner_calls = sum(r.get('n_boundaries', 0) for r in results)

    metrics = {
        'n_trajectories': n,
        'total_steps': total_steps,
        'total_correct': total_correct,
        'step_accuracy': step_accuracy,
        'type_accuracy': type_accuracy,
        'tsr': tsr,
        'tsr_count': tsr_count,
        'boundary_accuracy': boundary_correct / boundary_total if boundary_total > 0 else 0,
        'boundary_total': boundary_total,
        'within_phase_accuracy': within_correct / within_total if within_total > 0 else 0,
        'within_phase_total': within_total,
        'n_planner_calls': n_planner_calls,
        'avg_planner_calls_per_traj': n_planner_calls / n if n > 0 else 0,
    }

    # Length-stratified breakdown
    length_groups = defaultdict(list)
    for r in results:
        b = length_bucket(r['num_steps'])
        length_groups[b].append(r)

    length_metrics = {}
    for b, group in length_groups.items():
        g_steps = [s for r in group for s in r['step_results']]
        g_total = len(g_steps)
        g_correct = sum(1 for s in g_steps if s['extract_match'])
        g_type = sum(1 for s in g_steps if s['type_match'])
        g_tsr = sum(
            1 for r in group
            if all(s['extract_match'] for s in r['step_results']) and len(r['step_results']) == r['num_steps']
        )
        length_metrics[b] = {
            'n': len(group),
            'total_steps': g_total,
            'step_accuracy': g_correct / g_total if g_total > 0 else 0,
            'type_accuracy': g_type / g_total if g_total > 0 else 0,
            'tsr': g_tsr / len(group) if group else 0,
        }
    metrics['length_bucket_stats'] = length_metrics

    return metrics


def compute_second_half_metrics(results, split_key='midpoint'):
    """Compute StepAcc specifically for the second half of trajectories.

    Each result must have `split_key` (the step where second half starts)
    and `step_results` with `step_num`.
    """
    total = 0
    correct = 0
    for r in results:
        mid = r.get(split_key, r['num_steps'] // 2)
        for s in r['step_results']:
            if s['step_num'] >= mid:
                total += 1
                correct += int(s['extract_match'])
    return {
        'second_half_steps': total,
        'second_half_correct': correct,
        'second_half_accuracy': correct / total if total > 0 else 0,
    }


# ── I/O Helpers ──

def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    try:
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data, path):
    """Save dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def save_jsonl(data, path):
    """Save list of dicts to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=_json_default) + '\n')


def append_jsonl(item, path, lock=None):
    """Thread-safe append to JSONL file."""
    line = json.dumps(item, ensure_ascii=False, default=_json_default) + '\n'
    if lock:
        with lock:
            with open(path, 'a') as f:
                f.write(line)
    else:
        with open(path, 'a') as f:
            f.write(line)
