"""V22: Memory-Augmented Multi-Angle Reasoning evaluation.

Main eval script supporting:
  - Individual runs: memory_goal, memory_procedural, memory_type, memory_visual,
    angle_what_type, angle_where, angle_when, angle_why
  - Ensemble runs: ensemble_memory, ensemble_angle, ensemble_full

Uses pred history mode. Reuses consensus aggregation from v19_step_aware/eval_aisap.py.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# Reuse from v13
from v13_gui_360.eval_gui360_template import (
    parse_tool_call,
    _format_action_for_history,
    _encode_screenshot,
    SUPPORTED_ACTIONS,
)
from v13_gui_360.reward import compute_step_reward

# V22 modules
from v22_memory_agent.prompts import get_prompt, build_memory_prefix
from v22_memory_agent.memory_retrieval import (
    MemoryIndex,
    retrieve_goal,
    retrieve_procedural,
    retrieve_type,
    retrieve_visual,
    retrieve_random,
)
from v22_memory_agent.build_memory_index import extract_app_from_screenshot, normalize_action_type


# ═══════════════════════════════════════════════════════════════════════
# Experiment configurations
# ═══════════════════════════════════════════════════════════════════════

# Individual: each defines a list of 1 agent config
# Ensemble: each defines a list of N agent configs + 1 greedy anchor

EXPERIMENT_CONFIGS = {
    # --- Dim 1: Memory types (standard angle) ---
    "memory_goal":       [{"angle": "standard", "memory": "goal",       "temperature": 0.0}],
    "memory_procedural": [{"angle": "standard", "memory": "procedural", "temperature": 0.0}],
    "memory_type":       [{"angle": "standard", "memory": "type",       "temperature": 0.0}],
    "memory_visual":     [{"angle": "standard", "memory": "visual",     "temperature": 0.0}],

    # --- Dim 3: Reasoning angles (no memory) ---
    "angle_what_type": [{"angle": "what_type", "memory": None, "temperature": 0.0}],
    "angle_where":     [{"angle": "where",     "memory": None, "temperature": 0.0}],
    "angle_when":      [{"angle": "when",      "memory": None, "temperature": 0.0}],
    "angle_why":       [{"angle": "why",        "memory": None, "temperature": 0.0}],

    # --- Ensemble: 4 memory agents + greedy anchor ---
    "ensemble_memory": [
        {"angle": "standard", "memory": "goal",       "temperature": 0.7},
        {"angle": "standard", "memory": "procedural", "temperature": 0.7},
        {"angle": "standard", "memory": "type",       "temperature": 0.7},
        {"angle": "standard", "memory": "visual",     "temperature": 0.7},
        {"angle": "standard", "memory": None,          "temperature": 0.0},  # greedy anchor
    ],

    # --- Ensemble: 4 angle agents + greedy anchor ---
    "ensemble_angle": [
        {"angle": "what_type", "memory": None, "temperature": 0.7},
        {"angle": "where",     "memory": None, "temperature": 0.7},
        {"angle": "when",      "memory": None, "temperature": 0.7},
        {"angle": "why",        "memory": None, "temperature": 0.7},
        {"angle": "standard",  "memory": None, "temperature": 0.0},  # greedy anchor
    ],

    # --- Ensemble: 4 memory+angle combos + greedy anchor ---
    "ensemble_full": [
        {"angle": "what_type", "memory": "goal",       "temperature": 0.7},
        {"angle": "where",     "memory": "procedural", "temperature": 0.7},
        {"angle": "when",      "memory": "type",       "temperature": 0.7},
        {"angle": "why",        "memory": "visual",     "temperature": 0.7},
        {"angle": "standard",  "memory": None,          "temperature": 0.0},  # greedy anchor
    ],

    # --- Ablations: isolate memory content vs template effect ---
    # Random memory (GUIDED template + irrelevant examples) — controls for template
    "ablation_random_memory": [{"angle": "standard", "memory": "random", "temperature": 0.0}],
    # Best memory re-run with GT leak fix
    "memory_type_fixed":     [{"angle": "standard", "memory": "type",   "temperature": 0.0}],
    "memory_visual_fixed":   [{"angle": "standard", "memory": "visual", "temperature": 0.0}],
    "memory_goal_fixed":     [{"angle": "standard", "memory": "goal",   "temperature": 0.0}],

    # --- V22b: Intra-episode working memory ablation ---
    "working_reasoning":   [{"angle": "standard", "memory": None, "temperature": 0.0, "working_memory": "reasoning"}],
    "working_observation": [{"angle": "standard", "memory": None, "temperature": 0.0, "working_memory": "observation"}],
    "working_subtask":     [{"angle": "standard", "memory": None, "temperature": 0.0, "working_memory": "subtask"}],

    # --- V22c: AgentProg-inspired STP + Belief State ---
    "agentprog_stp":    [{"angle": "standard", "memory": None, "temperature": 0.0, "working_memory": "agentprog_stp"}],
    "agentprog_belief": [{"angle": "standard", "memory": None, "temperature": 0.0, "working_memory": "agentprog_belief"}],
    "agentprog_full":   [{"angle": "standard", "memory": None, "temperature": 0.0, "working_memory": "agentprog_full"}],
}


# ═══════════════════════════════════════════════════════════════════════
# Consensus aggregation (reused from v19_step_aware/eval_aisap.py)
# ═══════════════════════════════════════════════════════════════════════

def _action_type(action: Optional[Dict]) -> Optional[str]:
    if action is None:
        return None
    return action.get("action", "unknown")


def _get_coord(action: Optional[Dict]) -> Optional[Tuple[float, float]]:
    if action is None:
        return None
    coord = action.get("coordinate")
    if coord and len(coord) >= 2 and coord[0] is not None and coord[1] is not None:
        try:
            return (float(coord[0]), float(coord[1]))
        except (TypeError, ValueError):
            return None
    return None


def _coord_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def _cluster_actions(actions: List[Dict], merge_radius: float = 80.0) -> List[List[Dict]]:
    """Cluster actions by type, then merge nearby coordinates."""
    if not actions:
        return []

    type_groups = defaultdict(list)
    for a in actions:
        atype = _action_type(a)
        type_groups[atype].append(a)

    all_clusters = []
    for atype, group in type_groups.items():
        if atype == "type":
            text_groups = defaultdict(list)
            for a in group:
                text_key = str(a.get("text", ""))[:30].lower().strip()
                text_groups[text_key].append(a)
            for tg in text_groups.values():
                all_clusters.append(tg)
        else:
            remaining = list(group)
            while remaining:
                seed = remaining.pop(0)
                cluster = [seed]
                seed_coord = _get_coord(seed)
                still_remaining = []
                for a in remaining:
                    a_coord = _get_coord(a)
                    if seed_coord and a_coord and _coord_distance(seed_coord, a_coord) <= merge_radius:
                        cluster.append(a)
                    elif seed_coord is None and a_coord is None:
                        cluster.append(a)
                    else:
                        still_remaining.append(a)
                remaining = still_remaining
                all_clusters.append(cluster)

    return all_clusters


def _compute_centroid(actions: List[Dict]) -> Dict:
    """Compute centroid action from a cluster of similar actions."""
    if not actions:
        return {}

    type_counts = Counter(_action_type(a) for a in actions)
    best_type = type_counts.most_common(1)[0][0]
    result = {"action": best_type}

    if best_type in ("click", "long_press"):
        coords = [a.get("coordinate") for a in actions if a.get("coordinate")]
        coords = [c for c in coords if c and len(c) >= 2 and c[0] is not None and c[1] is not None]
        if coords:
            avg_x = sum(float(c[0]) for c in coords) / len(coords)
            avg_y = sum(float(c[1]) for c in coords) / len(coords)
            result["coordinate"] = [int(round(avg_x)), int(round(avg_y))]

    elif best_type == "type":
        coords = [a.get("coordinate") for a in actions if a.get("coordinate")]
        coords = [c for c in coords if c and len(c) >= 2 and c[0] is not None and c[1] is not None]
        if coords:
            avg_x = sum(float(c[0]) for c in coords) / len(coords)
            avg_y = sum(float(c[1]) for c in coords) / len(coords)
            result["coordinate"] = [int(round(avg_x)), int(round(avg_y))]
        texts = [str(a.get("text", "")) for a in actions if a.get("text")]
        if texts:
            result["text"] = Counter(texts).most_common(1)[0][0]

    elif best_type in ("swipe", "drag"):
        starts = [a.get("coordinate") for a in actions if a.get("coordinate")]
        starts = [c for c in starts if c and len(c) >= 2 and c[0] is not None]
        ends = [a.get("endCoordinate") for a in actions if a.get("endCoordinate")]
        ends = [c for c in ends if c and len(c) >= 2 and c[0] is not None]
        if starts:
            result["coordinate"] = [
                int(round(sum(float(c[0]) for c in starts) / len(starts))),
                int(round(sum(float(c[1]) for c in starts) / len(starts))),
            ]
        if ends:
            result["endCoordinate"] = [
                int(round(sum(float(c[0]) for c in ends) / len(ends))),
                int(round(sum(float(c[1]) for c in ends) / len(ends))),
            ]

    return result


def aggregate_consensus(
    samples: List[Dict[str, Any]],
    merge_radius: float = 80.0,
) -> Tuple[Optional[Dict], float, Dict]:
    """Aggregate N samples into a consensus action with optional similarity weighting."""
    N = len(samples)
    if N == 0:
        return None, 0.0, {}

    valid = [s for s in samples if s.get("action") is not None]
    valid_actions = [s["action"] for s in valid]
    if not valid_actions:
        return None, 0.0, {"n_valid": 0, "n_total": N}

    clusters = _cluster_actions(valid_actions, merge_radius)
    if not clusters:
        return None, 0.0, {"n_valid": len(valid_actions), "n_total": N, "n_clusters": 0}

    # Weight clusters: sum similarity scores of members
    # Build action -> sample lookup for similarity weighting
    action_sim = {}
    for s in valid:
        action_id = id(s["action"])
        action_sim[action_id] = s.get("similarity", 1.0)

    cluster_weights = []
    for cluster in clusters:
        weight = sum(action_sim.get(id(a), 1.0) for a in cluster)
        cluster_weights.append((cluster, weight, len(cluster)))

    # Sort by weight (similarity-weighted), break ties by size
    cluster_weights.sort(key=lambda x: (-x[1], -x[2]))
    best_cluster = cluster_weights[0][0]
    consensus_score = len(best_cluster) / N

    consensus_action = _compute_centroid(best_cluster)

    details = {
        "n_valid": len(valid_actions),
        "n_total": N,
        "n_clusters": len(clusters),
        "best_cluster_size": len(best_cluster),
        "best_cluster_weight": cluster_weights[0][1],
        "cluster_sizes": [len(c) for c, _, _ in cluster_weights],
    }

    return consensus_action, consensus_score, details


# ═══════════════════════════════════════════════════════════════════════
# Memory retrieval dispatch
# ═══════════════════════════════════════════════════════════════════════

def retrieve_memory(
    memory_index: MemoryIndex,
    memory_type: str,
    query_goal: str,
    query_app: str,
    query_step_idx: int,
    query_num_steps: int,
    query_action_sequence: List[str],
    query_action_type: str = "click",
    k: int = 3,
    exclude_episode_id: Optional[int] = None,
) -> List[Dict]:
    """Dispatch to the right retrieval function based on memory_type."""
    if memory_type == "goal":
        return retrieve_goal(memory_index, query_goal, query_step_idx, k, exclude_episode_id)
    elif memory_type == "procedural":
        return retrieve_procedural(memory_index, query_action_sequence, query_step_idx, k, exclude_episode_id)
    elif memory_type == "type":
        return retrieve_type(memory_index, query_app, query_action_type, query_step_idx, k, exclude_episode_id)
    elif memory_type == "visual":
        return retrieve_visual(memory_index, query_goal, query_app, query_step_idx, query_num_steps, k, exclude_episode_id)
    elif memory_type == "random":
        # Deterministic seed per (episode, step) for reproducibility
        seed = hash((exclude_episode_id or 0, query_step_idx)) % (2**31)
        return retrieve_random(memory_index, query_step_idx, k, exclude_episode_id, seed=seed)
    else:
        return []


# ═══════════════════════════════════════════════════════════════════════
# V22b: Working memory helpers
# ═══════════════════════════════════════════════════════════════════════

def extract_reasoning(pred_text: str, max_len: int = 200) -> str:
    """Extract reasoning text before <tool_call> tag.

    Returns the model's reasoning/thinking before its action output,
    truncated to max_len characters.
    """
    if not pred_text:
        return ""
    m = re.search(r'<tool_call>', pred_text)
    if m:
        reasoning = pred_text[:m.start()].strip()
    else:
        reasoning = pred_text.strip()
    if len(reasoning) > max_len:
        reasoning = reasoning[:max_len].rsplit(' ', 1)[0] + "..."
    return reasoning


def parse_subtask_list(reasoning: str) -> List[str]:
    """Parse numbered list (1. xxx  2. xxx) from reasoning text.

    Returns list of subtask strings, or empty list if no numbered list found.
    """
    if not reasoning:
        return []
    # Match lines starting with digits followed by . or )
    pattern = r'^\s*(\d+)[.)]\s*(.+)$'
    subtasks = []
    for line in reasoning.split('\n'):
        m = re.match(pattern, line.strip())
        if m:
            subtasks.append(m.group(2).strip())
    return subtasks


def extract_belief_state(pred_text: str) -> str:
    """Extract --- Belief State --- section from model output.

    Looks for content between '--- Belief State ---' and the next '---' header
    or <tool_call>. Falls back to extracting bullet points from reasoning.
    """
    if not pred_text:
        return ""
    # Try structured extraction
    m = re.search(
        r'---\s*Belief State\s*---\s*\n(.*?)(?=\n---|\n<tool_call>|$)',
        pred_text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # Fallback: extract bullet points from reasoning (before <tool_call>)
    tc = re.search(r'<tool_call>', pred_text)
    text = pred_text[:tc.start()] if tc else pred_text
    bullets = [line.strip() for line in text.split('\n')
               if line.strip().startswith(('-', '*', '\u2022'))]
    return '\n'.join(bullets) if bullets else ""


def update_belief_state(old_belief: str, new_belief: str, max_len: int = 500) -> str:
    """Merge new belief into old. Keep most recent bullets if over max_len."""
    if not new_belief:
        return old_belief
    if not old_belief:
        combined = new_belief
    else:
        combined = old_belief.rstrip('\n') + '\n' + new_belief
    if len(combined) <= max_len:
        return combined
    # Keep most recent lines (drop oldest first)
    lines = combined.split('\n')
    result = []
    total = 0
    for line in reversed(lines):
        if total + len(line) + 1 > max_len:
            break
        result.append(line)
        total += len(line) + 1
    return '\n'.join(reversed(result))


def advance_program_counter(reasoning: str, current_pc: int, workflow_len: int) -> int:
    """Advance program counter. Default +1. Hold if reasoning signals retry/incomplete.

    Holds if the reasoning contains signals like "retry", "not done", "still need",
    "try again", "failed", "incorrect". Otherwise advances by 1, capped at workflow_len - 1.
    """
    if not reasoning:
        return min(current_pc + 1, max(workflow_len - 1, 0))
    lower = reasoning.lower()
    hold_signals = ['retry', 'not done', 'still need', 'try again',
                    'failed', 'incorrect', 'not yet', 'repeat']
    for signal in hold_signals:
        if signal in lower:
            return current_pc
    return min(current_pc + 1, max(workflow_len - 1, 0))


def format_workflow_with_pc(workflow: list, pc: int) -> str:
    """Format numbered workflow with '# <-- current step' at pc position."""
    if not workflow:
        return ""
    lines = []
    for idx, step in enumerate(workflow):
        marker = "  # <-- current step" if idx == pc else ""
        lines.append(f"{idx + 1}. {step}{marker}")
    return '\n'.join(lines)


def format_working_memory(
    buffer: List[Dict],
    subtask_plan: Optional[List[str]],
    wm_type: Optional[str],
    current_step: int,
    program_counter: int = 0,
    belief_state: str = "",
) -> str:
    """Format accumulated buffer into prompt injection text.

    Args:
        buffer: list of dicts with keys: step_idx, reasoning
        subtask_plan: list of subtask strings (for subtask/agentprog types)
        wm_type: one of "reasoning", "observation", "subtask",
            "agentprog_stp", "agentprog_belief", "agentprog_full", or None
        current_step: current step index
        program_counter: current STP program counter (for agentprog_stp/full)
        belief_state: accumulated belief state string (for agentprog_belief/full)

    Returns:
        Formatted working memory string to inject into prompt.
    """
    if not wm_type or (not buffer and wm_type not in ("subtask", "agentprog_stp", "agentprog_belief", "agentprog_full")):
        return ""

    if wm_type == "reasoning":
        # Last 3 steps' reasoning
        recent = buffer[-3:]
        if not recent:
            return ""
        lines = ["Your reasoning from recent steps:"]
        for entry in recent:
            lines.append(f"Step {entry['step_idx'] + 1}: {entry['reasoning']}")
        return "\n".join(lines)

    elif wm_type == "observation":
        # Last 3 steps' reasoning, truncated to ~100 chars for factual observations
        recent = buffer[-3:]
        if not recent:
            return ""
        lines = ["Your observations from previous steps:"]
        for entry in recent:
            obs = entry["reasoning"]
            if len(obs) > 100:
                obs = obs[:100].rsplit(' ', 1)[0] + "..."
            lines.append(f"Step {entry['step_idx'] + 1}: {obs}")
        return "\n".join(lines)

    elif wm_type == "subtask":
        if not subtask_plan:
            return ""
        lines = ["Your task plan:"]
        for idx, subtask in enumerate(subtask_plan):
            step_num = idx + 1
            if step_num < current_step + 1:
                marker = "[DONE]"
            elif step_num == current_step + 1:
                marker = "[CURRENT]"
            else:
                marker = ""
            prefix = f"{step_num}. {marker} " if marker else f"{step_num}. "
            lines.append(f"{prefix}{subtask}")
        return "\n".join(lines)

    elif wm_type == "agentprog_stp":
        # Workflow with program counter marker
        if not subtask_plan:
            return ""
        return format_workflow_with_pc(subtask_plan, program_counter)

    elif wm_type == "agentprog_belief":
        # Accumulated belief state
        if not belief_state:
            return ""
        return f"Your accumulated belief state from previous steps:\n{belief_state}"

    elif wm_type == "agentprog_full":
        # Belief state only (workflow is passed separately via kwargs)
        if not belief_state:
            return ""
        return f"Your accumulated belief state from previous steps:\n{belief_state}"

    return ""


# ═══════════════════════════════════════════════════════════════════════
# Single agent call
# ═══════════════════════════════════════════════════════════════════════

def call_agent(
    client: OpenAI,
    model_name: str,
    b64_image: str,
    prompt_text: str,
    temperature: float = 0.0,
) -> Tuple[Optional[Dict], str]:
    """Call the model with a prompt and return (parsed_action, raw_text)."""
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
        {"type": "text", "text": prompt_text},
    ]}]

    pred_text = ""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
        )
        pred_text = response.choices[0].message.content or ""
    except Exception as e:
        print(f"  API error: {e}")

    pred_action = parse_tool_call(pred_text)
    if pred_action is None:
        m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
        if m:
            try:
                pred_action = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    return pred_action, pred_text


# ═══════════════════════════════════════════════════════════════════════
# Episode evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    agent_configs: List[Dict],
    memory_index: Optional[MemoryIndex],
    gt_history: bool = False,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
    k_memory: int = 3,
) -> Dict[str, Any]:
    """Evaluate one episode with memory-augmented multi-angle agents.

    For single-agent experiments (len(agent_configs)==1), runs the one agent per step.
    For ensembles, runs all agents per step and aggregates via consensus.
    """
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)
    is_ensemble = len(agent_configs) > 1

    # Extract app from first step
    app = "unknown"
    if steps:
        app = extract_app_from_screenshot(steps[0]["screenshot"])

    history = []
    action_type_sequence = []  # for procedural memory
    step_results = []
    first_error_step = None
    correct_steps = 0

    # V22b: working memory state
    working_memory_buffer = []  # grows across steps
    subtask_plan = None          # extracted at step 0 for subtask type

    # V22c: AgentProg state
    workflow = []         # from step 0 decomposition (STP/full types)
    program_counter = 0   # advances across steps (STP/full types)
    belief_state = ""     # accumulated belief (belief/full types)

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        history_text = "\n".join(history) if history else "None"
        b64 = _encode_screenshot(screenshot, image_max_pixels)

        # Determine action type for type-based retrieval (no GT leak)
        # Use previous step's predicted type; default to "click" (most common)
        current_action_type = action_type_sequence[-1] if action_type_sequence else "click"

        samples = []
        for cfg in agent_configs:
            # Retrieve memory if configured
            memory_prefix = ""
            similarity = 1.0
            if cfg.get("memory") and memory_index is not None:
                examples = retrieve_memory(
                    memory_index,
                    cfg["memory"],
                    query_goal=goal,
                    query_app=app,
                    query_step_idx=i,
                    query_num_steps=num_steps,
                    query_action_sequence=action_type_sequence,
                    query_action_type=current_action_type,
                    k=k_memory,
                    exclude_episode_id=episode_id,
                )
                memory_prefix = build_memory_prefix(examples)
                if examples:
                    similarity = sum(e.get("similarity", 0) for e in examples) / len(examples)

            # V22b/c: build working memory text
            wm_type = cfg.get("working_memory")
            wm_text = format_working_memory(
                working_memory_buffer, subtask_plan, wm_type, i,
                program_counter=program_counter, belief_state=belief_state,
            )

            # Build prompt (pass workflow for agentprog_full)
            prompt_kwargs = {}
            if wm_type == "agentprog_full" and workflow:
                prompt_kwargs["workflow"] = format_workflow_with_pc(workflow, program_counter)

            prompt_text = get_prompt(
                angle=cfg["angle"],
                instruction=goal,
                history=history_text,
                actions=SUPPORTED_ACTIONS,
                memory_prefix=memory_prefix,
                working_memory=wm_text,
                working_memory_type=wm_type or "",
                step_idx=i,
                **prompt_kwargs,
            )

            # Call model
            pred_action, pred_text = call_agent(
                client, model_name, b64, prompt_text, cfg["temperature"]
            )

            # V22b: extract reasoning and accumulate working memory
            if wm_type:
                reasoning = extract_reasoning(pred_text)
                if wm_type == "subtask" and i == 0:
                    subtask_plan = parse_subtask_list(reasoning)

                # V22c: AgentProg post-step updates
                if wm_type in ("agentprog_stp", "agentprog_full") and i == 0:
                    workflow = parse_subtask_list(reasoning)
                    subtask_plan = workflow  # reuse for format_working_memory
                    program_counter = 0

                if wm_type in ("agentprog_stp", "agentprog_full") and i > 0:
                    program_counter = advance_program_counter(
                        reasoning, program_counter, len(workflow)
                    )

                if wm_type in ("agentprog_belief", "agentprog_full"):
                    new_bs = extract_belief_state(pred_text)
                    belief_state = update_belief_state(belief_state, new_bs)

                working_memory_buffer.append({
                    "step_idx": i,
                    "reasoning": reasoning,
                })

            samples.append({
                "action": pred_action,
                "text": pred_text[:500],
                "angle": cfg["angle"],
                "memory": cfg.get("memory"),
                "temperature": cfg["temperature"],
                "similarity": similarity,
            })

        # Aggregate
        if is_ensemble:
            consensus_action, consensus_score, agg_details = aggregate_consensus(samples)
            pred_action_final = consensus_action
        else:
            pred_action_final = samples[0]["action"]
            consensus_score = 1.0
            agg_details = {"n_total": 1, "n_valid": 1 if pred_action_final else 0}

        # Score against GT
        if pred_action_final:
            fake_text = f"<action>{json.dumps(pred_action_final)}</action>"
        else:
            fake_text = ""
        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        success = reward >= match_threshold

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "pred_action": pred_action_final,
            "consensus_score": consensus_score,
            "n_agents": len(agent_configs),
            "agg_details": agg_details,
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
        })

        # Update history (pred history mode by default)
        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(pred_action_final, i + 1))

        # Track action types for procedural memory
        action_type_sequence.append(normalize_action_type(
            pred_action_final if pred_action_final else gt_action
        ))

        if success:
            correct_steps += 1
        elif first_error_step is None:
            first_error_step = i + 1

    progress = (first_error_step - 1) / num_steps if first_error_step else 1.0
    task_success = first_error_step is None and len(step_results) == num_steps

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": num_steps,
        "steps_evaluated": len(step_results),
        "correct_steps": correct_steps,
        "task_success": task_success,
        "progress": progress,
        "first_error_step": first_error_step,
        "steps": step_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# Breakdown analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_breakdown(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute per-step-position and per-task-length metrics."""
    step_pos_correct = defaultdict(int)
    step_pos_total = defaultdict(int)
    step_pos_type_correct = defaultdict(int)
    step_pos_consensus = defaultdict(list)

    length_buckets = {"1": (1, 1), "2-3": (2, 3), "4-5": (4, 5), "6+": (6, 999)}
    length_success = defaultdict(int)
    length_total = defaultdict(int)
    length_progress = defaultdict(float)
    length_steps_correct = defaultdict(int)
    length_steps_total = defaultdict(int)

    consensus_correct = []
    consensus_wrong = []

    for eid, result in results.items():
        num_steps = result["num_steps"]
        bucket = "6+"
        for bname, (lo, hi) in length_buckets.items():
            if lo <= num_steps <= hi:
                bucket = bname
                break

        length_total[bucket] += 1
        if result["task_success"]:
            length_success[bucket] += 1
        length_progress[bucket] += result["progress"]

        for step in result["steps"]:
            idx = step["step_idx"]
            step_pos_total[idx] += 1
            cs = step.get("consensus_score", 0)
            step_pos_consensus[idx].append(cs)
            if step["success"]:
                step_pos_correct[idx] += 1
                consensus_correct.append(cs)
            else:
                consensus_wrong.append(cs)
            if step.get("type_reward", 0) >= 1.0:
                step_pos_type_correct[idx] += 1
            length_steps_total[bucket] += 1
            if step["success"]:
                length_steps_correct[bucket] += 1

    # Format per-step-position
    max_step = max(step_pos_total.keys()) if step_pos_total else 0
    step_position_acc = {}
    for idx in range(min(max_step + 1, 15)):
        total = step_pos_total.get(idx, 0)
        correct = step_pos_correct.get(idx, 0)
        type_correct = step_pos_type_correct.get(idx, 0)
        cs_list = step_pos_consensus.get(idx, [])
        if total > 0:
            step_position_acc[f"step_{idx}"] = {
                "accuracy": correct / total,
                "type_accuracy": type_correct / total,
                "avg_consensus": sum(cs_list) / len(cs_list) if cs_list else 0,
                "total": total,
                "correct": correct,
            }

    # Format per-task-length
    task_length_metrics = {}
    for bname in length_buckets:
        total = length_total.get(bname, 0)
        if total > 0:
            s_total = length_steps_total.get(bname, 0)
            task_length_metrics[bname] = {
                "tsr": length_success[bname] / total,
                "avg_progress": length_progress[bname] / total,
                "step_sr": length_steps_correct[bname] / s_total if s_total > 0 else 0,
                "num_episodes": total,
                "num_success": length_success[bname],
            }

    # Consensus analysis
    consensus_analysis = {}
    if consensus_correct and consensus_wrong:
        consensus_analysis = {
            "avg_consensus_correct": sum(consensus_correct) / len(consensus_correct),
            "avg_consensus_wrong": sum(consensus_wrong) / len(consensus_wrong),
            "n_correct": len(consensus_correct),
            "n_wrong": len(consensus_wrong),
        }

    return {
        "step_position_accuracy": step_position_acc,
        "task_length_metrics": task_length_metrics,
        "consensus_analysis": consensus_analysis,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V22: Memory-Augmented Multi-Angle Reasoning")
    parser.add_argument("--experiment", required=True,
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help="Experiment configuration to run")
    parser.add_argument("--test_data", required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", default="v22_memory_agent/outputs")
    parser.add_argument("--index_dir", default="v22_memory_agent/indices",
                        help="Path to memory index directory")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--k_memory", type=int, default=3,
                        help="Number of memory examples to retrieve per step")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    agent_configs = EXPERIMENT_CONFIGS[args.experiment]
    is_ensemble = len(agent_configs) > 1
    uses_memory = any(cfg["memory"] is not None for cfg in agent_configs)

    print(f"V22 Experiment: {args.experiment}")
    print(f"  Agents: {len(agent_configs)} ({'ensemble' if is_ensemble else 'single'})")
    print(f"  Memory: {uses_memory}")
    print(f"  GT history: {args.gt_history}")
    for i, cfg in enumerate(agent_configs):
        print(f"    Agent {i}: angle={cfg['angle']}, memory={cfg['memory']}, temp={cfg['temperature']}")

    # Load test data
    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    total_loaded = len(episodes)
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes (shard [{args.start}:{args.end}] of {total_loaded})")

    # Load memory index if needed
    memory_index = None
    if uses_memory:
        print(f"Loading memory index from {args.index_dir} ...")
        memory_index = MemoryIndex(args.index_dir)
        # Eagerly load to catch errors early
        n_steps = len(memory_index.step_records)
        n_episodes = len(memory_index.episode_metadata)
        print(f"  Memory index: {n_episodes} episodes, {n_steps} steps")

    # Setup output
    exp_output_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(exp_output_dir, exist_ok=True)

    client = OpenAI(base_url=args.api_url, api_key="dummy")

    results = {}
    total_success = 0
    total_progress = 0.0
    total_steps = 0
    total_correct = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode, client, args.model_name, ep,
                agent_configs, memory_index,
                args.gt_history, args.match_threshold,
                args.image_max_pixels, args.k_memory,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc=f"V22 {args.experiment}")
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                eid = futures[future]
                print(f"\nError in episode {eid}: {e}")
                continue

            eid = result["episode_id"]
            results[str(eid)] = result

            if result["task_success"]:
                total_success += 1
            total_progress += result["progress"]
            total_steps += result["steps_evaluated"]
            total_correct += result["correct_steps"]

            n = len(results)
            pbar.update(1)
            pbar.set_postfix({
                "TSR": f"{total_success/n:.3f}",
                "StepSR": f"{total_correct/total_steps:.3f}" if total_steps > 0 else "0",
            })
        pbar.close()

    # Compute breakdowns
    breakdown = compute_breakdown(results)

    n = len(results)
    summary = {
        "experiment": args.experiment,
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "total_steps_evaluated": total_steps,
        "total_steps_correct": total_correct,
        "agents_per_step": len(agent_configs),
        "total_inference_calls": total_steps * len(agent_configs),
        "gt_history": args.gt_history,
        "match_threshold": args.match_threshold,
        "k_memory": args.k_memory,
        **breakdown,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(exp_output_dir, f"summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(exp_output_dir, f"results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print(f"\n{'='*65}")
    print(f"V22 Results: {args.experiment}")
    print(f"{'='*65}")
    print(f"  TSR:              {summary['tsr']*100:.1f}%")
    print(f"  Step SR:          {summary['step_sr']*100:.1f}%")
    print(f"  Avg Progress:     {summary['avg_progress']*100:.1f}%")
    print(f"  Agents/step:      {len(agent_configs)}")
    print(f"  Total inferences: {summary['total_inference_calls']}")
    hist_mode = "GT history" if args.gt_history else "Pred history"
    print(f"  History mode:     {hist_mode}")
    print(f"  (Baselines: standard=21.9%, type_focused=23.6%, BoN-5=26.6%)")

    # Consensus analysis
    ca = summary.get("consensus_analysis", {})
    if ca:
        print(f"\n  --- Consensus Analysis ---")
        print(f"  Correct steps avg consensus: {ca.get('avg_consensus_correct', 0)*100:.1f}%")
        print(f"  Wrong steps avg consensus:   {ca.get('avg_consensus_wrong', 0)*100:.1f}%")

    print(f"\n  --- By Task Length ---")
    for bname, metrics in sorted(summary.get("task_length_metrics", {}).items()):
        print(f"  {bname:>5s} steps: TSR={metrics['tsr']*100:5.1f}%  "
              f"StepSR={metrics['step_sr']*100:5.1f}%  "
              f"Progress={metrics['avg_progress']*100:5.1f}%  "
              f"(n={metrics['num_episodes']})")

    print(f"\n  --- By Step Position ---")
    for sname, metrics in sorted(summary.get("step_position_accuracy", {}).items(),
                                  key=lambda x: int(x[0].split("_")[1])):
        print(f"  {sname:>8s}: acc={metrics['accuracy']*100:5.1f}%  "
              f"type_acc={metrics['type_accuracy']*100:5.1f}%  "
              f"consensus={metrics.get('avg_consensus', 0)*100:5.1f}%  "
              f"(n={metrics['total']})")

    print(f"\n  Output: {exp_output_dir}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
