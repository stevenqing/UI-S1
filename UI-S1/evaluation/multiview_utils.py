"""
Multi-View evaluation utilities — Variants A/B/C/D/E/F.

Variant A: Augmented System Prompt (append expert analysis section to MOBILE_USE)
Variant B: Dedicated Aggregator Prompt (replace system prompt with D2-style prompt)
Variant C: State Document Accumulation (accumulate view analyses across steps)
Variant D: Hybrid (B's prompt + C's accumulated summary)
Variant E: Action-Level Voting (3 independent passes, majority vote)
Variant F: Ultra-Concise Recommendation (one-line structured hint, ~150 chars)
"""

import copy
import re
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# ── View System Prompts ───────────────────────────────────────────────────

VISUAL_ANALYST_PROMPT = """\
You are a Visual Analyst for a mobile GUI agent. Analyze the CURRENT screenshot and describe:
1. Screen Layout: type of screen (home, settings, search results, dialog, etc.)
2. Key Interactive Elements: buttons, text fields, icons, toggles with approximate positions
3. Current State: loading, scrolled position, keyboard visible, popup shown
4. Task-Relevant Elements: which elements relate to the given task

Provide concise structured text. Do NOT output any action tags or JSON."""

TASK_TRACKER_PROMPT = """\
You are a Task Progress Tracker for a mobile GUI agent. Analyze task progress:
1. Task Goal: restate the user's objective
2. Completed Steps: summarize what's been done based on action history
3. Current Position: does the current screen match expected progress?
4. Next Sub-goal: what should be accomplished next?
5. Potential Issues: signs of failed actions or being off-track?

Provide concise structured text. Do NOT output any action tags or JSON."""

# ── Aggregation Templates ─────────────────────────────────────────────────

AGGREGATION_CONTEXT_TEMPLATE = """\
## Expert Analyses for Current Step

### Visual Analysis:
{visual_analysis}

### Task Progress Analysis:
{task_analysis}

Based on the above analyses and the screenshot, decide the next action."""

# Variant A: Append to existing MOBILE_USE system prompt
SYSTEM_PROMPT_SUFFIX_A = """
## Expert Analyses

Before each screenshot, you will receive analyses from two specialists:
- **Visual Analysis**: Current screen layout, interactive elements, and state
- **Task Progress Analysis**: Task completion status, next sub-goal, potential issues

Use these analyses to inform your decision. They provide complementary perspectives
on the current state. Make your final action decision based on both the analyses
and the screenshot."""

# Variant B: Dedicated aggregator system prompt (replaces MOBILE_USE entirely)
AGGREGATOR_SYSTEM_PROMPT_B = """\
You are a GUI agent with access to expert analyses. You are given a task and your action history, with screenshots, and analyses from two specialist systems.

## Input Structure
At each step you receive:
1. Expert analyses (Visual Analysis + Task Progress Analysis)
2. Current screenshot
3. Your previous actions (in conversation history)

## How to Use Expert Analyses
- **Visual Analysis** tells you what UI elements are present, what screen you're on, and what's interactive. Use this to accurately identify targets.
- **Task Progress Analysis** tells you what has been accomplished, what the next sub-goal is, and whether you might be off-track. Use this to choose the RIGHT action.
- If the analyses disagree with what you see, trust the screenshot but note the discrepancy.

## Output Format

{output_format}

## Action Space

{action_space}

## Note
- Plan the task and explain your reasoning step-by-step in `think` part.
- Reference the expert analyses in your reasoning when relevant.
- Write your action in the `action` part according to the action space."""

# Variant C/D: State document injection template
STATE_DOC_TEMPLATE = """\
## State Tracker Notes
{state_document}
"""

ACCUMULATED_CONTEXT_TEMPLATE = """\
## State Tracker Notes (recent history)
{accumulated_summary}

## Expert Analyses for Current Step

### Visual Analysis:
{visual_analysis}

### Task Progress Analysis:
{task_analysis}

Based on the above context and the screenshot, decide the next action."""

# ── Variant E: Voting prompts (3 different perspectives, each outputs action) ──

VOTING_PROMPT_VISUAL = """\
You are a GUI agent. Focus on the VISUAL LAYOUT of the current screen. \
Identify the correct UI element to interact with based on its visual appearance and position.

## Output Format

{output_format}

## Action Space

{action_space}

## Note
- Focus on what you SEE on screen: buttons, icons, text fields, their positions.
- Write your action in the `action` part according to the action space."""

VOTING_PROMPT_TASK = """\
You are a GUI agent. Focus on TASK PLANNING. \
Think about what step is needed next to accomplish the goal, then find the right element.

## Output Format

{output_format}

## Action Space

{action_space}

## Note
- Focus on the GOAL and what logical step comes next.
- Write your action in the `action` part according to the action space."""

# ── Variant F: Ultra-concise view prompts ──────────────────────────────────

CONCISE_VISUAL_PROMPT = """\
You are a GUI screen reader. Output EXACTLY one line describing the current screen.
Format: "Screen: [type]. Elements: [key element 1] at ([x1],[y1]), [key element 2] at ([x2],[y2])."
Output ONLY the one-line description, nothing else."""

CONCISE_TASK_PROMPT = """\
You are a task progress checker. Output EXACTLY one line.
Format: "Goal: [goal]. Status: [done/in-progress/off-track]. Next: [next action needed]."
Output ONLY the one-line description, nothing else."""

CONCISE_INJECTION_TEMPLATE = "Context: {visual_hint} | {task_hint}"

# ── Variant G: D2-style single-turn state tracker ────────────────────────

OBSERVER_PROMPT_G = """\
You are a State Tracker for a mobile GUI agent. Your job is NOT to describe what the screenshot looks like — the action model can see the screenshot itself. Instead, track task-relevant state changes.

Task: {task_goal}

{last_action_section}

Report ONLY these items (be very concise, one line each):
1. LOCATION: What app view/screen is currently showing
2. CHANGE: What specifically changed after the last action (skip if first step)
3. PROGRESS: Which sub-steps of the task are done, which remain
4. WARNING: Any signs of being stuck (repeated actions, unexpected state) — write "none" if everything looks normal

Do NOT describe UI elements the model can see in the screenshot. Focus on WHAT CHANGED and WHERE WE ARE in the task."""

ACTION_WITH_STATE_DOC_PROMPT_G = """\
You are a helpful assistant controlling a mobile device. Given a screenshot of the current screen, user instruction, and a state tracker's notes, you need to decide the next action to take.

The instruction is:
{task_goal}

State tracker notes:
{state_document}

Use the state tracker's notes to understand:
- What has already been accomplished (don't repeat completed sub-steps)
- What the immediate next sub-step should be
- Whether we're in the right place to perform the next action

## Output Format

{output_format}

## Action Space

{action_space}

## Note
- Plan the task and explain your reasoning step-by-step in `think` part.
- Write your action in the `action` part according to the action space."""

# GUI-360 variant: uses <tool_call> format
ACTION_WITH_OBSERVER_PROMPT_GUI360 = """\
You are a helpful assistant. Given a screenshot of the current screen, user instruction, and a state tracker's notes, you need to decide the next action to take.

State tracker's analysis of the current screen:
{observer_notes}

Use the state tracker's analysis to understand:
- What screen/view is currently showing
- Which UI elements are task-relevant
- What the immediate next action should be

Then output your action within <tool_call></tool_call> tag."""


# ── Core API Call ──────────────────────────────────────────────────────────


def call_model_with_limit(messages, model_name, api_url, max_tokens=None,
                          max_retries=3):
    """Call vLLM server via OpenAI-compatible API."""
    client = OpenAI(api_key="EMPTY", base_url=api_url, timeout=600)

    kwargs = {"extra_body": {"top_k": 1}}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return ""


# ── Helper Functions ───────────────────────────────────────────────────────


def _strip_action_tags(text):
    """Remove <think>, <action>, <tool_call>, and code-block wrappers from
    view analysis text so they don't confuse the aggregator's output parser."""
    if not text:
        return text
    text = re.sub(r'</?(?:think|action|tool_call)>', '', text)
    text = re.sub(r'^```\w*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    return text.strip()


def _ensure_content_list(content):
    """Normalise content to a list of dicts."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, str):
                out.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                out.append(item)
        return out
    return content


def _extract_images_from_content(content):
    """Extract image items from content list."""
    content = _ensure_content_list(content)
    return [item for item in content if isinstance(item, dict) and (
        item.get("type") == "image_url" or item.get("type") == "image"
        or "image_url" in item or "image" in item
    )]


def _extract_text_from_content(content):
    """Extract all text from content list, concatenated."""
    content = _ensure_content_list(content)
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item["text"])
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def _extract_last_user_msg(messages):
    """Return deep copy of the last user message."""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return copy.deepcopy(msg)
    return None


def _inject_text_before_last_image(msgs, inject_text):
    """Inject text block before the last image in the last user message."""
    last_user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return msgs

    content = _ensure_content_list(msgs[last_user_idx]["content"])

    last_img_pos = None
    for j in range(len(content) - 1, -1, -1):
        if isinstance(content[j], dict) and (
            content[j].get("type") == "image_url"
            or content[j].get("type") == "image"
            or "image" in content[j]
            or "image_url" in content[j]
        ):
            last_img_pos = j
            break

    inject = {"type": "text", "text": inject_text}
    if last_img_pos is not None:
        content.insert(last_img_pos, inject)
    else:
        content.append(inject)

    msgs[last_user_idx]["content"] = content
    return msgs


def _modify_system_prompt(msgs, modifier_fn):
    """Apply modifier_fn to system prompt in messages (in-place)."""
    for msg in msgs:
        if msg["role"] == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = modifier_fn(item["text"])
                        return msgs
                    elif isinstance(item, dict) and "text" in item:
                        item["text"] = modifier_fn(item["text"])
                        return msgs
            elif isinstance(content, str):
                msg["content"] = modifier_fn(content)
                return msgs
    return msgs


def _replace_system_prompt(msgs, new_prompt):
    """Replace the system prompt entirely."""
    for msg in msgs:
        if msg["role"] == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = new_prompt
                        return msgs
                    elif isinstance(item, dict) and "text" in item:
                        item["text"] = new_prompt
                        return msgs
            elif isinstance(content, str):
                msg["content"] = new_prompt
                return msgs
    return msgs


def _get_system_prompt_text(msgs):
    """Extract system prompt text from messages."""
    for msg in msgs:
        if msg["role"] == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        return item["text"]
            elif isinstance(content, str):
                return content
    return ""


# ── View Calling ──────────────────────────────────────────────────────────


def _call_views_stateless(base_messages, model_name, api_url, executor=None):
    """Call View 1 (Visual Analyst) and View 2 (Task Tracker) statelessly.

    Each view gets a single-turn call with the screenshot + task context.
    """
    last_user = _extract_last_user_msg(base_messages)
    if last_user is None:
        return "", ""

    images = _extract_images_from_content(last_user.get("content", []))
    text = _extract_text_from_content(last_user.get("content", []))

    # Build view messages: system + user with task text + images
    def _build_view_msg(view_prompt):
        content = [{"type": "text", "text": f"Task context:\n{text[:1000]}\n\nAnalyze the current screenshot below:"}]
        content.extend(copy.deepcopy(images))
        return [
            {"role": "system", "content": view_prompt},
            {"role": "user", "content": content},
        ]

    v1_msgs = _build_view_msg(VISUAL_ANALYST_PROMPT)
    v2_msgs = _build_view_msg(TASK_TRACKER_PROMPT)

    if executor is not None:
        fut1 = executor.submit(
            call_model_with_limit, v1_msgs, model_name, api_url, 512)
        fut2 = executor.submit(
            call_model_with_limit, v2_msgs, model_name, api_url, 512)
        visual_analysis = fut1.result()
        task_analysis = fut2.result()
    else:
        visual_analysis = call_model_with_limit(
            v1_msgs, model_name, api_url, 512)
        task_analysis = call_model_with_limit(
            v2_msgs, model_name, api_url, 512)

    return _strip_action_tags(visual_analysis), _strip_action_tags(task_analysis)


def _call_views_with_context(base_messages, model_name, api_url,
                              view1_history, view2_history,
                              last_action_summary=None, step_count=0,
                              executor=None):
    """Call views with persistent per-view conversation context.

    Updates view1_history and view2_history in-place.
    Returns (visual_analysis, task_analysis).
    """
    last_user = _extract_last_user_msg(base_messages)
    if last_user is None:
        return "", ""

    images = _extract_images_from_content(last_user.get("content", []))
    text = _extract_text_from_content(last_user.get("content", []))

    # Build user message for views
    task_text = f"Task: {text[:800]}"
    if step_count > 0 and last_action_summary:
        task_text += f"\n\nLast action taken by agent (step {step_count}):\n{last_action_summary}"
    task_text += "\n\nAnalyze the current screenshot below:"

    view_user_content = [{"type": "text", "text": task_text}]
    view_user_content.extend(copy.deepcopy(images))
    view_user_msg = {"role": "user", "content": view_user_content}

    view1_history.append(copy.deepcopy(view_user_msg))
    view2_history.append(copy.deepcopy(view_user_msg))

    # Keep only last 2 images in each view history
    v1_msgs = _slim_view_messages(view1_history, max_images=2)
    v2_msgs = _slim_view_messages(view2_history, max_images=2)

    if executor is not None:
        fut1 = executor.submit(
            call_model_with_limit, v1_msgs, model_name, api_url, 512)
        fut2 = executor.submit(
            call_model_with_limit, v2_msgs, model_name, api_url, 512)
        visual_analysis = fut1.result()
        task_analysis = fut2.result()
    else:
        visual_analysis = call_model_with_limit(
            v1_msgs, model_name, api_url, 512)
        task_analysis = call_model_with_limit(
            v2_msgs, model_name, api_url, 512)

    # Store responses in histories
    view1_history.append({"role": "assistant", "content": visual_analysis})
    view2_history.append({"role": "assistant", "content": task_analysis})

    return _strip_action_tags(visual_analysis), _strip_action_tags(task_analysis)


def _slim_view_messages(messages, max_images=2):
    """Keep only the last N images in a view's message history."""
    msgs = copy.deepcopy(messages)
    image_positions = []
    for i, msg in enumerate(msgs):
        if msg["role"] != "user":
            continue
        content = _ensure_content_list(msg.get("content", []))
        for j, item in enumerate(content):
            if isinstance(item, dict) and (
                item.get("type") == "image_url" or item.get("type") == "image"
                or "image_url" in item or "image" in item
            ):
                image_positions.append((i, j))

    to_remove = len(image_positions) - max_images
    if to_remove > 0:
        for msg_idx, content_idx in image_positions[:to_remove]:
            content = _ensure_content_list(msgs[msg_idx].get("content", []))
            content[content_idx] = {"type": "text", "text": "[previous screenshot omitted]"}
            msgs[msg_idx]["content"] = content

    return msgs


# ── Variant-Specific Aggregator Builders ───────────────────────────────────


def build_aggregator_variant_a(base_messages, visual_analysis, task_analysis):
    """Variant A: Append expert analysis section to system prompt + inject in user msg."""
    msgs = copy.deepcopy(base_messages)

    # Augment system prompt
    _modify_system_prompt(msgs, lambda sp: sp + SYSTEM_PROMPT_SUFFIX_A)

    # Inject analyses before last image in user message
    clean_visual = visual_analysis or "(no visual analysis available)"
    clean_task = task_analysis or "(no task analysis available)"
    context_text = AGGREGATION_CONTEXT_TEMPLATE.format(
        visual_analysis=clean_visual, task_analysis=clean_task)
    _inject_text_before_last_image(msgs, context_text)

    return msgs


def build_aggregator_variant_b(base_messages, visual_analysis, task_analysis,
                                output_format="", action_space=""):
    """Variant B: Replace system prompt with dedicated aggregator prompt + inject in user msg."""
    msgs = copy.deepcopy(base_messages)

    # Extract output_format and action_space from existing system prompt if not provided
    if not output_format or not action_space:
        sys_text = _get_system_prompt_text(msgs)
        # Try to extract from existing MOBILE_USE format
        if "## Output Format" in sys_text and "## Action Space" in sys_text:
            parts = sys_text.split("## Action Space")
            action_space = parts[1].split("## Note")[0].strip() if len(parts) > 1 else action_space
            fmt_parts = sys_text.split("## Output Format")
            if len(fmt_parts) > 1:
                output_format = fmt_parts[1].split("## Action Space")[0].strip()

    new_prompt = AGGREGATOR_SYSTEM_PROMPT_B.format(
        output_format=output_format, action_space=action_space)
    _replace_system_prompt(msgs, new_prompt)

    # Inject analyses before last image
    clean_visual = visual_analysis or "(no visual analysis available)"
    clean_task = task_analysis or "(no task analysis available)"
    context_text = AGGREGATION_CONTEXT_TEMPLATE.format(
        visual_analysis=clean_visual, task_analysis=clean_task)
    _inject_text_before_last_image(msgs, context_text)

    return msgs


def build_aggregator_variant_c(base_messages, visual_analysis, task_analysis,
                                state_document_entries, step_num):
    """Variant C: Accumulate state document + inject in user msg.

    System prompt is augmented with Variant A's suffix.
    state_document_entries is modified in-place (appended to).
    """
    msgs = copy.deepcopy(base_messages)

    # Augment system prompt (same as A)
    _modify_system_prompt(msgs, lambda sp: sp + SYSTEM_PROMPT_SUFFIX_A)

    # Build compact state document entry from current analyses
    v_compact = (visual_analysis or "")[:150].replace("\n", " ")
    t_compact = (task_analysis or "")[:150].replace("\n", " ")
    entry = f"Step {step_num}: Visual: {v_compact} | Progress: {t_compact}"
    state_document_entries.append(entry)

    # Use last 5 entries as state doc
    recent = state_document_entries[-5:]
    state_doc = "\n".join(recent)

    # Inject state doc + current analyses
    clean_visual = visual_analysis or "(no visual analysis available)"
    clean_task = task_analysis or "(no task analysis available)"
    context_text = ACCUMULATED_CONTEXT_TEMPLATE.format(
        accumulated_summary=state_doc,
        visual_analysis=clean_visual,
        task_analysis=clean_task)
    _inject_text_before_last_image(msgs, context_text)

    return msgs


def build_aggregator_variant_d(base_messages, visual_analysis, task_analysis,
                                state_document_entries, step_num,
                                output_format="", action_space=""):
    """Variant D: Hybrid — B's dedicated prompt + C's accumulated summary."""
    msgs = copy.deepcopy(base_messages)

    # Replace system prompt (like B)
    if not output_format or not action_space:
        sys_text = _get_system_prompt_text(msgs)
        if "## Output Format" in sys_text and "## Action Space" in sys_text:
            parts = sys_text.split("## Action Space")
            action_space = parts[1].split("## Note")[0].strip() if len(parts) > 1 else action_space
            fmt_parts = sys_text.split("## Output Format")
            if len(fmt_parts) > 1:
                output_format = fmt_parts[1].split("## Action Space")[0].strip()

    new_prompt = AGGREGATOR_SYSTEM_PROMPT_B.format(
        output_format=output_format, action_space=action_space)
    _replace_system_prompt(msgs, new_prompt)

    # Accumulate state document (like C)
    v_compact = (visual_analysis or "")[:150].replace("\n", " ")
    t_compact = (task_analysis or "")[:150].replace("\n", " ")
    entry = f"Step {step_num}: Visual: {v_compact} | Progress: {t_compact}"
    state_document_entries.append(entry)

    recent = state_document_entries[-5:]
    state_doc = "\n".join(recent)

    # Inject accumulated + current analyses
    clean_visual = visual_analysis or "(no visual analysis available)"
    clean_task = task_analysis or "(no task analysis available)"
    context_text = ACCUMULATED_CONTEXT_TEMPLATE.format(
        accumulated_summary=state_doc,
        visual_analysis=clean_visual,
        task_analysis=clean_task)
    _inject_text_before_last_image(msgs, context_text)

    return msgs


# ── Variant E: Voting ──────────────────────────────────────────────────────


def _parse_action_from_response(response):
    """Extract action dict from model response (handles both AC and GUI-360 formats)."""
    if not response:
        return None
    # Try <action>...</action> format (AC)
    m = re.search(r'<action>\s*(.*?)\s*</action>', response, re.DOTALL)
    if m:
        try:
            import json
            return json.loads(m.group(1))
        except Exception:
            return None
    # Try <tool_call>...</tool_call> format (GUI-360)
    m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
    if m:
        try:
            import json
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


def _majority_vote_actions(responses):
    """Given list of (response_text, parsed_action), pick majority action type
    and return the full response from the winner."""
    import json
    from collections import Counter

    parsed = []
    for resp in responses:
        action = _parse_action_from_response(resp)
        parsed.append((resp, action))

    # Vote on action type
    type_votes = Counter()
    for resp, action in parsed:
        if action and "action" in action:
            type_votes[action["action"]] += 1
        elif action and "function" in action:
            type_votes[action["function"]] += 1

    if not type_votes:
        # No valid actions, return first response
        return responses[0] if responses else "", "no_valid_actions"

    winner_type = type_votes.most_common(1)[0][0]

    # Among responses with winning type, pick the first one
    for resp, action in parsed:
        if action:
            atype = action.get("action") or action.get("function", "")
            if atype == winner_type:
                return resp, f"vote:{dict(type_votes)}"

    return responses[0] if responses else "", f"fallback:{dict(type_votes)}"


def variant_e_voting_step(base_messages, model_name, api_url, executor=None):
    """Variant E: Run 3 independent forward passes with different prompt biases,
    then majority vote on the action.

    Pass 1: Original prompt (unchanged)
    Pass 2: Visual-focus prompt
    Pass 3: Task-focus prompt
    """
    # Extract output_format and action_space from existing system prompt
    sys_text = _get_system_prompt_text(base_messages)
    output_format = ""
    action_space = ""
    if "## Output Format" in sys_text and "## Action Space" in sys_text:
        parts = sys_text.split("## Action Space")
        action_space = parts[1].split("## Note")[0].strip() if len(parts) > 1 else ""
        fmt_parts = sys_text.split("## Output Format")
        if len(fmt_parts) > 1:
            output_format = fmt_parts[1].split("## Action Space")[0].strip()

    # Pass 1: Original (no modification)
    msgs_original = copy.deepcopy(base_messages)

    # Pass 2: Visual-focus prompt
    msgs_visual = copy.deepcopy(base_messages)
    visual_prompt = VOTING_PROMPT_VISUAL.format(
        output_format=output_format, action_space=action_space)
    _replace_system_prompt(msgs_visual, visual_prompt)

    # Pass 3: Task-focus prompt
    msgs_task = copy.deepcopy(base_messages)
    task_prompt = VOTING_PROMPT_TASK.format(
        output_format=output_format, action_space=action_space)
    _replace_system_prompt(msgs_task, task_prompt)

    if executor is not None:
        fut1 = executor.submit(
            call_model_with_limit, msgs_original, model_name, api_url, None)
        fut2 = executor.submit(
            call_model_with_limit, msgs_visual, model_name, api_url, None)
        fut3 = executor.submit(
            call_model_with_limit, msgs_task, model_name, api_url, None)
        r1 = fut1.result()
        r2 = fut2.result()
        r3 = fut3.result()
    else:
        r1 = call_model_with_limit(msgs_original, model_name, api_url)
        r2 = call_model_with_limit(msgs_visual, model_name, api_url)
        r3 = call_model_with_limit(msgs_task, model_name, api_url)

    winner_response, vote_info = _majority_vote_actions([r1, r2, r3])

    debug_info = {
        "visual_analysis": f"Pass1(original): {r1[:200]}",
        "task_analysis": f"Pass2(visual): {r2[:200]} | Pass3(task): {r3[:200]}",
        "variant": "E",
        "vote_info": vote_info,
    }
    return winner_response, debug_info


# ── Variant F: Ultra-Concise ──────────────────────────────────────────────


def _call_concise_views(base_messages, model_name, api_url, executor=None):
    """Call ultra-concise views. Each produces ~1 line, max_tokens=80."""
    last_user = _extract_last_user_msg(base_messages)
    if last_user is None:
        return "", ""

    images = _extract_images_from_content(last_user.get("content", []))
    text = _extract_text_from_content(last_user.get("content", []))

    def _build(prompt):
        content = [{"type": "text", "text": f"Task: {text[:500]}"}]
        content.extend(copy.deepcopy(images))
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]

    v1_msgs = _build(CONCISE_VISUAL_PROMPT)
    v2_msgs = _build(CONCISE_TASK_PROMPT)

    if executor is not None:
        fut1 = executor.submit(
            call_model_with_limit, v1_msgs, model_name, api_url, 80)
        fut2 = executor.submit(
            call_model_with_limit, v2_msgs, model_name, api_url, 80)
        visual_hint = fut1.result()
        task_hint = fut2.result()
    else:
        visual_hint = call_model_with_limit(v1_msgs, model_name, api_url, 80)
        task_hint = call_model_with_limit(v2_msgs, model_name, api_url, 80)

    # Clean: take only first line, strip tags
    visual_hint = _strip_action_tags(visual_hint).split('\n')[0][:200]
    task_hint = _strip_action_tags(task_hint).split('\n')[0][:200]
    return visual_hint, task_hint


def build_aggregator_variant_f(base_messages, visual_hint, task_hint):
    """Variant F: Inject ultra-concise one-line context. No system prompt change."""
    msgs = copy.deepcopy(base_messages)

    # Only inject if we got actual content
    if visual_hint or task_hint:
        context = CONCISE_INJECTION_TEMPLATE.format(
            visual_hint=visual_hint or "N/A",
            task_hint=task_hint or "N/A")
        _inject_text_before_last_image(msgs, context)

    return msgs


# ── MultiViewPipeline (trajectory-level, for AC eval) ─────────────────────


class MultiViewPipeline:
    """Maintains state across trajectory steps for multi-view evaluation.

    Supports variants A/B/C/D via the `variant` parameter.
    - A/B: Views are stateless (single-turn per step)
    - C/D: Views are stateless + state document accumulation
    - All variants: Aggregator uses full multi-turn conversation history
    """

    def __init__(self, model_name, api_url, variant="B"):
        self.model_name = model_name
        self.api_url = api_url
        self.variant = variant.upper()

        # State document for variants C/D
        self.state_document_entries = []
        self._step_count = 0
        self._last_action_summary = None

        # Per-view conversation context (for variants A/B with context)
        self.view1_history = [
            {"role": "system", "content": VISUAL_ANALYST_PROMPT}
        ]
        self.view2_history = [
            {"role": "system", "content": TASK_TRACKER_PROMPT}
        ]

    def step(self, base_messages, executor=None):
        """Execute one multi-view step.

        Args:
            base_messages: OpenAI-format messages (system + history + current user).
            executor: Optional ThreadPoolExecutor for parallel view calls.

        Returns:
            (aggregator_response, debug_info)
        """
        self._step_count += 1

        # Variant E: completely different flow (voting, no analysis injection)
        if self.variant == "E":
            return variant_e_voting_step(
                base_messages, self.model_name, self.api_url, executor=executor)

        # Variant F: ultra-concise views
        if self.variant == "F":
            visual_hint, task_hint = _call_concise_views(
                base_messages, self.model_name, self.api_url, executor=executor)
            agg_msgs = build_aggregator_variant_f(
                base_messages, visual_hint, task_hint)
            aggregator_response = call_model_with_limit(
                agg_msgs, self.model_name, self.api_url, max_tokens=None)
            debug_info = {
                "visual_analysis": visual_hint,
                "task_analysis": task_hint,
                "variant": self.variant,
            }
            return aggregator_response, debug_info

        # Variants A/B/C/D: call views then build aggregator
        visual_analysis, task_analysis = _call_views_stateless(
            base_messages, self.model_name, self.api_url, executor=executor)

        # Build aggregator messages based on variant
        if self.variant == "A":
            agg_msgs = build_aggregator_variant_a(
                base_messages, visual_analysis, task_analysis)
        elif self.variant == "B":
            agg_msgs = build_aggregator_variant_b(
                base_messages, visual_analysis, task_analysis)
        elif self.variant == "C":
            agg_msgs = build_aggregator_variant_c(
                base_messages, visual_analysis, task_analysis,
                self.state_document_entries, self._step_count)
        elif self.variant == "D":
            agg_msgs = build_aggregator_variant_d(
                base_messages, visual_analysis, task_analysis,
                self.state_document_entries, self._step_count)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        # Call aggregator
        aggregator_response = call_model_with_limit(
            agg_msgs, self.model_name, self.api_url, max_tokens=None)

        # Save action summary for next step
        self._last_action_summary = aggregator_response[:300] \
            if aggregator_response else None

        debug_info = {
            "visual_analysis": visual_analysis,
            "task_analysis": task_analysis,
            "variant": self.variant,
        }
        return aggregator_response, debug_info


# ── Stateless convenience (for single-step evals like GUI-360) ────────────


def multiview_step_stateless(base_messages, model_name, api_url,
                              variant="B", executor=None):
    """Single-step multi-view (no persistent context). For step-level evals."""
    if variant == "G":
        return variant_g_step_gui360(
            base_messages, model_name, api_url, executor=executor)
    pipeline = MultiViewPipeline(model_name, api_url, variant=variant)
    return pipeline.step(base_messages, executor=executor)


def variant_g_step_gui360(base_messages, model_name, api_url, executor=None):
    """Variant G for GUI-360: single-step with dedicated observer + action prompt.

    1. Observer: structured analysis (LOCATION/elements/task-relevant)
    2. Action model: new single-turn call with observer notes + screenshot
       Uses a dedicated prompt instead of injecting into existing system prompt.
    """
    last_user = _extract_last_user_msg(base_messages)
    if last_user is None:
        return "", {}

    images = _extract_images_from_content(last_user.get("content", []))
    text = _extract_text_from_content(last_user.get("content", []))

    # 1. Call Observer (single-turn, same as other variants)
    obs_prompt = OBSERVER_PROMPT_G.format(
        task_goal=text[:800],
        last_action_section="This is a single-step evaluation. No previous actions.",
    )
    obs_content = [{"type": "text", "text": obs_prompt}]
    obs_content.extend(copy.deepcopy(images))
    obs_msgs = [{"role": "user", "content": obs_content}]

    observer_desc = call_model_with_limit(
        obs_msgs, model_name, api_url, max_tokens=256)
    observer_desc = _strip_action_tags(observer_desc)

    # 2. Build action model messages — new single-turn with dedicated prompt
    # Extract original system prompt content (for action space/format info)
    sys_text = _get_system_prompt_text(base_messages)

    action_prompt = ACTION_WITH_OBSERVER_PROMPT_GUI360.format(
        observer_notes=observer_desc or "(no observation)",
    )

    # Keep the original system prompt but prepend observer context
    combined_system = action_prompt + "\n\n" + sys_text

    # Build single-turn: system (combined) + user (original content)
    action_msgs = [
        {"role": "system", "content": combined_system},
        copy.deepcopy(last_user),
    ]

    action_response = call_model_with_limit(
        action_msgs, model_name, api_url, max_tokens=None)

    debug_info = {
        "visual_analysis": observer_desc,
        "task_analysis": "G-style single-turn",
        "variant": "G",
    }
    return action_response, debug_info


# ── Variant G Pipeline (D2-style single-turn state tracker) ──────────────


class VariantGPipeline:
    """D2-style single-turn state tracker for trajectory evaluation.

    Key differences from A-F:
    - Completely bypasses multi-turn history (gen_next_round)
    - Each step = 2 single-turn calls: observer + action model
    - Observer output accumulated into state document
    - Action model gets screenshot + state document (not conversation history)
    """

    def __init__(self, model_name, api_url, output_format="", action_space=""):
        self.model_name = model_name
        self.api_url = api_url
        self.output_format = output_format
        self.action_space = action_space
        self.state_document_entries = []
        self._step_count = 0
        self._last_pred_action_desc = None

    def step(self, task_goal, image_content, executor=None):
        """Execute one step of D2-style evaluation.

        Args:
            task_goal: Task instruction string.
            image_content: List of image content items (OpenAI format dicts
                with type=image_url).
            executor: Optional ThreadPoolExecutor for parallel calls.

        Returns:
            (action_response, debug_info)
        """
        self._step_count += 1
        step_num = self._step_count

        # --- Build last_action_section for Observer ---
        if step_num == 1:
            last_action_section = "This is the first step. No previous actions have been taken."
        elif self._last_pred_action_desc:
            last_action_section = (
                f"Previous step result:\n{self._last_pred_action_desc}\n"
                f"The above action was taken in step {step_num - 1}. "
                f"Look at the screenshot to see what actually happened."
            )
        else:
            last_action_section = f"Previous action (step {step_num - 1}) could not be parsed."

        # --- 1. Call Observer (single-turn) ---
        obs_prompt = OBSERVER_PROMPT_G.format(
            task_goal=task_goal,
            last_action_section=last_action_section,
        )
        obs_content = [{"type": "text", "text": obs_prompt}]
        obs_content.extend(copy.deepcopy(image_content))
        obs_msgs = [{"role": "user", "content": obs_content}]

        observer_desc = call_model_with_limit(
            obs_msgs, self.model_name, self.api_url, max_tokens=256)
        observer_desc = _strip_action_tags(observer_desc)

        # --- 2. Build state document ---
        state_entries_formatted = []
        for i, entry in enumerate(self.state_document_entries):
            state_entries_formatted.append(f"Step {i + 1}: {entry}")

        # Current observation
        if observer_desc:
            state_entries_formatted.append(
                f"Step {step_num} (current): {observer_desc.strip()}")

        if state_entries_formatted:
            state_doc_str = "\n".join(state_entries_formatted)
        else:
            state_doc_str = "(First step — no prior state)"

        # --- 3. Call Action Model (single-turn) ---
        action_prompt = ACTION_WITH_STATE_DOC_PROMPT_G.format(
            task_goal=task_goal,
            state_document=state_doc_str,
            output_format=self.output_format,
            action_space=self.action_space,
        )
        action_content = [{"type": "text", "text": action_prompt}]
        action_content.extend(copy.deepcopy(image_content))
        action_msgs = [{"role": "user", "content": action_content}]

        action_response = call_model_with_limit(
            action_msgs, self.model_name, self.api_url, max_tokens=None)

        # --- 4. Update state document ---
        # Extract action description for next step's observer
        action_desc = ""
        parsed = _parse_action_from_response(action_response)
        if parsed:
            action_type = parsed.get("action", parsed.get("function", ""))
            action_desc = f" → {action_type}"
            for k, v in parsed.items():
                if k in ("action", "function"):
                    continue
                if v is not None:
                    action_desc += f" {k}={v}"

        self._last_pred_action_desc = action_desc.strip() if action_desc else None

        # Compact observer desc for state doc (max 200 chars)
        obs_compact = observer_desc.strip()[:200] if observer_desc else "no observation"
        entry = f"{obs_compact}{action_desc}"
        self.state_document_entries.append(entry)

        debug_info = {
            "visual_analysis": observer_desc,
            "task_analysis": f"State doc ({len(self.state_document_entries)} entries)",
            "variant": "G",
            "state_document": state_doc_str[:500],
        }
        return action_response, debug_info
