"""AgentProg-inspired prompt templates (V22c).

Adapted from AgentProg (MobiSys 2026) two key mechanisms:
  1. Semantic Task Program (STP): structured workflow + program counter
  2. Belief State: incrementally updated hypothesis list about environment state

3 experiment variants:
  - agentprog_stp:    STP only (workflow decomposition + PC tracking)
  - agentprog_belief: Belief State only (observation + accumulated beliefs)
  - agentprog_full:   STP + Belief State combined
"""


# ---------------------------------------------------------------------------
# STP Step 0: decompose task into workflow, then execute step 1
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# STP Steps 1+: show workflow with program counter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Belief State: observation + belief + action every step
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Full Step 0: decompose + belief + act
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Full Steps 1+: workflow + belief + act
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------
AGENTPROG_TEMPLATES = {
    "agentprog_stp": AGENTPROG_STP_PROMPT,
    "agentprog_belief": AGENTPROG_BELIEF_PROMPT,
    "agentprog_full": AGENTPROG_FULL_PROMPT,
}

AGENTPROG_STEP0_TEMPLATES = {
    "agentprog_stp": AGENTPROG_STP_STEP0_PROMPT,
    "agentprog_full": AGENTPROG_FULL_STEP0_PROMPT,
}
