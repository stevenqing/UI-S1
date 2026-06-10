"""V22: Reasoning angle prompt templates, memory injection, and working memory.

4 reasoning angles (Dimension 3) + memory injection template.
V22b: 3 intra-episode working memory types (reasoning, observation, subtask).
V22c: AgentProg-inspired STP + Belief State (agentprog_stp, agentprog_belief, agentprog_full).
All follow the same base structure as v19 prompts.
"""


# ---------------------------------------------------------------------------
# Standard prompt (baseline / greedy anchor)
# ---------------------------------------------------------------------------
STANDARD_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# ---------------------------------------------------------------------------
# Angle 1: What-Type — focus on interaction type first
# Targets: 49.2% wrong-type errors
# ---------------------------------------------------------------------------
ANGLE_WHAT_TYPE = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Before acting, determine the interaction TYPE first:
1. What type of interaction is needed next? (click to select/navigate, type to enter text, drag to move/scroll, wheel_mouse_input to scroll)
2. Justify WHY this type and not another. For example: if a text field is already focused and waiting for input, the action must be "type", not "click".
3. Only THEN identify the target UI element and its precise coordinates.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# ---------------------------------------------------------------------------
# Angle 2: Where — focus on grounding the target UI element
# Targets: Grounding errors
# ---------------------------------------------------------------------------
ANGLE_WHERE = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Identify the target UI element systematically:
1. List the main UI elements visible on screen (buttons, text fields, menus, tabs, icons).
2. Which element is most relevant to the current sub-goal given the instruction and history?
3. Describe its exact position (top/bottom, left/right, near which landmark).
4. Estimate its pixel coordinates carefully based on the screenshot dimensions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# ---------------------------------------------------------------------------
# Angle 3: When — focus on progress tracking, avoid skipping ahead
# Targets: Jumping-ahead on long tasks
# ---------------------------------------------------------------------------
ANGLE_WHEN = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Review your progress carefully before acting:
1. What sub-steps does the overall instruction require?
2. Which sub-steps have already been completed based on the action history?
3. What is the IMMEDIATE next sub-step? Do NOT skip ahead — complete one step at a time.
4. Does the current screen state match what you expect after the previous actions?

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# ---------------------------------------------------------------------------
# Angle 4: Why — focus on purpose and sub-goal decomposition
# Targets: Planning bottleneck
# ---------------------------------------------------------------------------
ANGLE_WHY = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Explain the PURPOSE of your next action:
1. Break the overall goal into sub-goals (e.g., "open menu" → "select option" → "enter value" → "confirm").
2. Which sub-goal are you working on right now?
3. WHY is this specific action the right way to advance that sub-goal?
4. What do you expect the screen to look like AFTER this action?

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# ---------------------------------------------------------------------------
# Memory injection wrapper
# ---------------------------------------------------------------------------
MEMORY_PREFIX_TEMPLATE = """Based on similar past experiences:
{memory_examples}
Use these examples as reference. Adapt to the current screen and instruction."""


def format_memory_example(idx: int, goal: str, thought: str, action_summary: str) -> str:
    """Format a single retrieved memory example."""
    return f"Example {idx} (from task: \"{goal}\"): Reasoning: {thought} Action taken: {action_summary}"


def build_memory_prefix(examples: list) -> str:
    """Build the memory injection prefix from retrieved examples.

    Args:
        examples: list of dicts with keys: goal, thought, action_summary, similarity
    Returns:
        Formatted memory string to inject into prompt, or empty string if no examples.
    """
    if not examples:
        return ""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(format_memory_example(
            i,
            ex.get("goal", ""),
            ex.get("thought", "N/A"),
            ex.get("action_summary", ""),
        ))
    return MEMORY_PREFIX_TEMPLATE.format(memory_examples="\n".join(lines))


# ---------------------------------------------------------------------------
# Prompt with memory guidance (uses USER_PROMPT_TEMPLATE_GUIDED pattern)
# ---------------------------------------------------------------------------
GUIDED_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction, history of actions, and reference examples from past experience, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

{guidance}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Consider the reference examples along with the screenshot and history. Then decide the best action to take.

Output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# Angle prompts with memory guidance slot
ANGLE_WHAT_TYPE_GUIDED = """You are a helpful assistant. Given a screenshot of the current screen, user instruction, history of actions, and reference examples from past experience, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

{guidance}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Consider the reference examples. Then determine the interaction TYPE first:
1. What type of interaction is needed next? (click to select/navigate, type to enter text, drag to move/scroll, wheel_mouse_input to scroll)
2. Justify WHY this type and not another.
3. Only THEN identify the target UI element and its precise coordinates.

Output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


ANGLE_WHERE_GUIDED = """You are a helpful assistant. Given a screenshot of the current screen, user instruction, history of actions, and reference examples from past experience, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

{guidance}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Consider the reference examples. Then identify the target UI element systematically:
1. List the main UI elements visible on screen.
2. Which element is most relevant to the current sub-goal?
3. Describe its exact position and estimate pixel coordinates carefully.

Output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


ANGLE_WHEN_GUIDED = """You are a helpful assistant. Given a screenshot of the current screen, user instruction, history of actions, and reference examples from past experience, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

{guidance}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Consider the reference examples. Then review your progress carefully:
1. What sub-steps does the overall instruction require?
2. Which sub-steps have already been completed?
3. What is the IMMEDIATE next sub-step? Do NOT skip ahead.
4. Does the current screen state match expectations?

Output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


ANGLE_WHY_GUIDED = """You are a helpful assistant. Given a screenshot of the current screen, user instruction, history of actions, and reference examples from past experience, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

{guidance}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Consider the reference examples. Then explain the PURPOSE of your next action:
1. Break the overall goal into sub-goals.
2. Which sub-goal are you working on right now?
3. WHY is this specific action the right way to advance that sub-goal?
4. What do you expect the screen to look like AFTER this action?

Output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# ---------------------------------------------------------------------------
# V22b: Working memory prompts (intra-episode accumulated context)
# ---------------------------------------------------------------------------

WORKING_REASONING_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

{working_memory}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


WORKING_OBSERVATION_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

{working_memory}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


WORKING_SUBTASK_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

{working_memory}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


WORKING_SUBTASK_STEP0_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Before acting, decompose the instruction into numbered sub-steps. List each concrete UI action needed to accomplish the goal, then take the first step.

Format your plan as:
1. <first sub-step>
2. <second sub-step>
...

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# ---------------------------------------------------------------------------
# V22c: AgentProg-inspired STP + Belief State prompts
# ---------------------------------------------------------------------------

# STP Step 0: decompose task into workflow, then execute step 1
AGENTPROG_STP_STEP0_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Before acting, decompose the instruction into a structured workflow of concrete UI steps:
1. <step>
2. <step>
...

Now execute step 1. Analyze the screenshot and decide the action.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# STP Steps 1+: show workflow with program counter
AGENTPROG_STP_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

Your workflow for this task:
{working_memory}

You are executing the step marked "# <-- current step".
Analyze the screenshot to verify the current step matches what you see, then take the action.

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning—verify the current step against the screenshot, then decide the action.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# Belief State: observation + belief + action every step
AGENTPROG_BELIEF_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

{working_memory}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, observe the screenshot and describe what you see.
Then update your belief state — facts about task progress, including things not visible on screen.
Finally, decide the next action.

Format:
--- Observation ---
(what you see on the current screen)
--- Belief State ---
(bullet list of facts about task progress)
--- Action Reasoning ---
(why this action is correct)

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# Full Step 0: decompose + belief + act
AGENTPROG_FULL_STEP0_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Before acting, decompose the instruction into a structured workflow of concrete UI steps:
1. <step>
2. <step>
...

Then observe the screenshot, establish your initial belief state, and execute step 1.

Format:
--- Observation ---
(what you see on the current screen)
--- Belief State ---
(bullet list of initial facts)
--- Action Reasoning ---
(why this action for step 1)

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# Full Steps 1+: workflow + belief + act
AGENTPROG_FULL_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

Your workflow for this task:
{workflow}

{working_memory}

You are executing the step marked "# <-- current step".

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, observe the screenshot and verify the current step.
Then update your belief state with any new facts, including things not visible on screen.
Finally, decide the next action.

Format:
--- Observation ---
(what you see on the current screen)
--- Belief State ---
(bullet list of facts about task progress)
--- Action Reasoning ---
(why this action is correct for the current step)

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


WORKING_MEMORY_TEMPLATES = {
    "reasoning": WORKING_REASONING_PROMPT,
    "observation": WORKING_OBSERVATION_PROMPT,
    "subtask": WORKING_SUBTASK_PROMPT,
    "agentprog_stp": AGENTPROG_STP_PROMPT,
    "agentprog_belief": AGENTPROG_BELIEF_PROMPT,
    "agentprog_full": AGENTPROG_FULL_PROMPT,
}


# ---------------------------------------------------------------------------
# Registry: map experiment name → (prompt_template, uses_memory, uses_guidance)
# ---------------------------------------------------------------------------
ANGLE_TEMPLATES = {
    "standard":       STANDARD_PROMPT,
    "what_type":      ANGLE_WHAT_TYPE,
    "where":          ANGLE_WHERE,
    "when":           ANGLE_WHEN,
    "why":            ANGLE_WHY,
}

ANGLE_TEMPLATES_GUIDED = {
    "standard":       GUIDED_PROMPT,
    "what_type":      ANGLE_WHAT_TYPE_GUIDED,
    "where":          ANGLE_WHERE_GUIDED,
    "when":           ANGLE_WHEN_GUIDED,
    "why":            ANGLE_WHY_GUIDED,
}


def get_prompt(angle: str, instruction: str, history: str, actions: str,
               memory_prefix: str = "", working_memory: str = "",
               working_memory_type: str = "", step_idx: int = 0,
               **kwargs) -> str:
    """Build the final prompt string for a given angle and optional memory.

    Args:
        angle: one of "standard", "what_type", "where", "when", "why"
        instruction: task goal
        history: formatted action history
        actions: SUPPORTED_ACTIONS string
        memory_prefix: formatted memory examples (empty string = no memory)
        working_memory: formatted intra-episode working memory text
        working_memory_type: one of "reasoning", "observation", "subtask",
            "agentprog_stp", "agentprog_belief", "agentprog_full", or ""
        step_idx: current step index (used for subtask/STP step-0 variant)
        **kwargs: extra fields (e.g. workflow for agentprog_full)

    Returns:
        Fully formatted prompt string.
    """
    # V22b/c: working memory takes priority over inter-episode memory
    if working_memory_type:
        if working_memory_type == "subtask" and step_idx == 0:
            # Step 0: use special decomposition prompt (no working_memory slot)
            return WORKING_SUBTASK_STEP0_PROMPT.format(
                instruction=instruction,
                history=history,
                actions=actions,
            )

        # V22c: AgentProg step-0 variants
        if working_memory_type == "agentprog_stp" and step_idx == 0:
            return AGENTPROG_STP_STEP0_PROMPT.format(
                instruction=instruction,
                history=history,
                actions=actions,
            )
        if working_memory_type == "agentprog_full" and step_idx == 0:
            return AGENTPROG_FULL_STEP0_PROMPT.format(
                instruction=instruction,
                history=history,
                actions=actions,
            )

        # V22c: AgentProg full uses both workflow and working_memory slots
        if working_memory_type == "agentprog_full" and step_idx > 0:
            return AGENTPROG_FULL_PROMPT.format(
                instruction=instruction,
                history=history,
                workflow=kwargs.get("workflow", ""),
                working_memory=working_memory,
                actions=actions,
            )

        template = WORKING_MEMORY_TEMPLATES.get(working_memory_type, WORKING_REASONING_PROMPT)
        return template.format(
            instruction=instruction,
            history=history,
            working_memory=working_memory,
            actions=actions,
        )

    if memory_prefix:
        template = ANGLE_TEMPLATES_GUIDED.get(angle, GUIDED_PROMPT)
        return template.format(
            instruction=instruction,
            history=history,
            guidance=memory_prefix,
            actions=actions,
        )
    else:
        template = ANGLE_TEMPLATES.get(angle, STANDARD_PROMPT)
        return template.format(
            instruction=instruction,
            history=history,
            actions=actions,
        )
