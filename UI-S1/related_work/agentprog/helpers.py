"""AgentProg helper functions for STP and Belief State management.

Core mechanisms adapted from AgentProg (MobiSys 2026):
  - STP: program counter tracking, workflow formatting
  - Belief State: extraction, merging, and truncation
"""

import re
from typing import List, Optional


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


def format_agentprog_working_memory(
    wm_type: str,
    subtask_plan: Optional[List[str]],
    program_counter: int = 0,
    belief_state: str = "",
) -> str:
    """Format AgentProg working memory for prompt injection.

    Args:
        wm_type: one of "agentprog_stp", "agentprog_belief", "agentprog_full"
        subtask_plan: list of workflow step strings
        program_counter: current STP program counter
        belief_state: accumulated belief state string

    Returns:
        Formatted working memory string.
    """
    if wm_type == "agentprog_stp":
        if not subtask_plan:
            return ""
        return format_workflow_with_pc(subtask_plan, program_counter)

    elif wm_type == "agentprog_belief":
        if not belief_state:
            return ""
        return f"Your accumulated belief state from previous steps:\n{belief_state}"

    elif wm_type == "agentprog_full":
        # Belief state only (workflow is passed separately via kwargs)
        if not belief_state:
            return ""
        return f"Your accumulated belief state from previous steps:\n{belief_state}"

    return ""
