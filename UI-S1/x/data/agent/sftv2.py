"""
SFTv2Format: matches the GUI-360 SFT v2 training data format exactly.

Key differences from JsonFormat:
- Single-turn per step (not multi-turn accumulation)
- User message: "<image>\nYou are a helpful assistant..."
- History of previous actions flattened as text in the user message
- Output uses <tool_call> tags with nested {"function", "args", "status"} JSON
- No system prompt (everything in user message)
- App-specific action spaces (excel, ppt, word) loaded from sftv2_templates.json
"""

import copy
import json
import os
import re

from x.data.agent.base import STD_THINKING_KEY, BaseFormatAbs
from x.data.text import parse_tags
from x.qwen.image import make_qwen_image_item

# Load app-specific prompt templates extracted from gui360_train.json
# Templates use __INSTRUCTION__ and __HISTORY__ as placeholders
_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'sftv2_templates.json')
with open(_TEMPLATE_PATH) as _f:
    _APP_TEMPLATES = json.load(_f)

# Default to excel if app can't be determined
_DEFAULT_APP = 'excel'


def _detect_app(line):
    """Detect app type from execution_id (e.g., 'excel_1_81' -> 'excel')."""
    eid = line.get('execution_id', '')
    if eid:
        app = eid.split('_')[0].lower()
        if app in _APP_TEMPLATES:
            return app
    # Fallback: check screenshot path
    for step in line.get('steps', []):
        ss = step.get('screenshot', '')
        for app in _APP_TEMPLATES:
            if f'/{app}/' in ss.lower():
                return app
    return _DEFAULT_APP


def _build_user_text(line, history_text, app=None):
    """Build the user message text using the app-specific template."""
    if app is None:
        app = _detect_app(line)
    template = _APP_TEMPLATES.get(app, _APP_TEMPLATES[_DEFAULT_APP])
    return template.replace('__INSTRUCTION__', line['goal']).replace('__HISTORY__', history_text)


def _format_action_for_history(action_content, step_id, image_ele=None):
    """Format a flat action_content dict into a history string like SFT v2 training data."""
    action_type = action_content.get('action', '')

    if action_type == 'click':
        coord = action_content.get('coordinate', [0, 0])
        button = action_content.get('button', 'left')
        return f"Step {step_id}: click(coordinate={coord}, button='{button}', double=False)"

    elif action_type == 'type':
        coord = action_content.get('coordinate', [0, 0])
        text = action_content.get('text', '')
        keys = action_content.get('keys', '')
        if text:
            display_text = text[:30] + '...' if len(text) > 30 else text
            return f"Step {step_id}: type(coordinate={coord}, text='{display_text}')"
        elif keys:
            return f"Step {step_id}: type(coordinate={coord}, keys='{keys}')"
        else:
            return f"Step {step_id}: type(coordinate={coord})"

    elif action_type == 'drag':
        coord = action_content.get('coordinate', [0, 0])
        coord2 = action_content.get('coordinate2', [0, 0])
        return f"Step {step_id}: drag(start_coordinate={coord}, end_coordinate={coord2})"

    elif action_type == 'swipe':
        coord = action_content.get('coordinate', [0, 0])
        coord2 = action_content.get('coordinate2', [0, 0])
        return f"Step {step_id}: drag(start_coordinate={coord}, end_coordinate={coord2})"

    elif action_type == 'wheel_mouse_input':
        coord = action_content.get('coordinate', [0, 0])
        wheel_dist = action_content.get('wheel_dist', 0)
        return f"Step {step_id}: wheel_mouse_input(coordinate={coord}, wheel_dist={wheel_dist})"

    elif action_type in ('terminate', 'wait', ''):
        return f"Step {step_id}: Task completed (FINISH)"

    else:
        # Generic: include action type and all args (covers app-specific actions)
        args_str = ", ".join(f"{k}={v}" for k, v in action_content.items() if k != 'action' and v is not None)
        return f"Step {step_id}: {action_type}({args_str})"


def _flat_to_nested(action_content, image_ele):
    """
    Convert flat action_content (RL internal format) to nested SFT v2 format.
    flat: {"action": "click", "coordinate": [x,y], ...}
    nested: {"function": "click", "args": {"coordinate": [x,y], "button": "left"}, "status": "CONTINUE"}
    """
    action_type = action_content.get('action', '')

    if action_type in ('terminate', '', 'wait'):
        return {"function": "", "args": {}, "status": "FINISH"}

    args = {}

    if action_type == 'click':
        args['coordinate'] = action_content.get('coordinate', [0, 0])
        button = action_content.get('button', 'left')
        args['button'] = button or 'left'
        args['double'] = False

    elif action_type == 'type':
        args['coordinate'] = action_content.get('coordinate', [0, 0])
        if action_content.get('text'):
            args['text'] = action_content['text']
        if action_content.get('keys'):
            args['keys'] = action_content['keys']

    elif action_type == 'drag':
        args['start_coordinate'] = action_content.get('coordinate', [0, 0])
        args['end_coordinate'] = action_content.get('coordinate2', [0, 0])

    elif action_type == 'swipe':
        args['start_coordinate'] = action_content.get('coordinate', [0, 0])
        args['end_coordinate'] = action_content.get('coordinate2', [0, 0])
        return {"function": "drag", "args": args, "status": "CONTINUE"}

    elif action_type == 'wheel_mouse_input':
        args['coordinate'] = action_content.get('coordinate', [0, 0])
        args['wheel_dist'] = action_content.get('wheel_dist', 0)

    else:
        # Generic: pass through all fields as args (covers app-specific actions)
        for k, v in action_content.items():
            if k != 'action' and v is not None:
                args[k] = v

    return {"function": action_type, "args": args, "status": "CONTINUE"}


def _nested_to_flat(nested):
    """
    Convert nested SFT v2 format back to flat action_content (RL internal format).
    nested: {"function": "click", "args": {"coordinate": [x,y], "button": "left"}, "status": "CONTINUE"}
    flat: {"action": "click", "coordinate": [x,y], ...}
    """
    function = nested.get('function', '')
    args = nested.get('args', {})
    status = nested.get('status', 'CONTINUE')

    if not function or status == 'FINISH':
        return {"action": "terminate"}

    result = {"action": function}

    if function == 'click':
        result['coordinate'] = args.get('coordinate')
        if 'button' in args:
            result['button'] = args['button']

    elif function == 'type':
        result['coordinate'] = args.get('coordinate')
        if 'text' in args:
            result['text'] = args['text']
        if 'keys' in args:
            result['text'] = args['keys']  # map keys -> text for reward compatibility

    elif function == 'drag':
        result['coordinate'] = args.get('start_coordinate')
        result['coordinate2'] = args.get('end_coordinate')

    elif function == 'wheel_mouse_input':
        result['coordinate'] = args.get('coordinate')
        result['wheel_dist'] = args.get('wheel_dist', 0)

    else:
        # Generic: flatten args (covers app-specific actions like select_text, etc.)
        result.update(args)

    return result


class SFTv2Format(BaseFormatAbs):
    """
    Format class matching the SFT v2 training data exactly.

    - Single-turn per step (history flattened as text)
    - User msg: "<image>\nYou are a helpful assistant..."
    - Output: "<tool_call>\n{JSON}\n</tool_call>"
    - App-specific action spaces (excel, ppt, word)
    """

    def __init__(self, space_file):
        super().__init__(space_file, add_thought=False, force_add_thought=False)

    def format_action(self, action_content, image_ele):
        """Format a flat action_content dict to the nested SFT v2 JSON string."""
        action_content = self._format_action_base(action_content, image_ele)
        nested = _flat_to_nested(action_content, image_ele)
        return json.dumps(nested, ensure_ascii=False)

    def parse_action(self, action_str, restrict_mode=False):
        """Parse the nested JSON from inside <tool_call> tags back to flat format."""
        nested = json.loads(action_str)
        return _nested_to_flat(nested)

    def parse_response(self, model_response, restrict_mode=False):
        """
        Parse model response. Extracts <tool_call> content and any pre-tag text as 'think'.
        Returns dict with 'tool_call', 'think', 'action_content'.
        """
        result = parse_tags(model_response, ['tool_call'])

        # Extract any text before <tool_call> as pseudo-think
        tc_match = re.search(r'<tool_call>', model_response)
        if tc_match:
            pre_text = model_response[:tc_match.start()].strip()
            result['think'] = pre_text if pre_text else None
        else:
            result['think'] = None

        if result.get('tool_call'):
            action_content = self.parse_action(result['tool_call'], restrict_mode=restrict_mode)
            result['action_content'] = action_content
        else:
            result['action_content'] = None

        return result

    def format_response(self, step, image_ele, add_thought=True):
        """Format GT action as SFT v2 response: optional reasoning + <tool_call>JSON</tool_call>"""
        action_content = step['action_content']
        action_json_str = self.format_action(action_content, image_ele)

        response = ""
        # Include thought/reasoning as free text before tool_call (matching SFT v2 training)
        thought = step.get(STD_THINKING_KEY) or step.get('motivation', '')
        if add_thought and thought:
            response += f"Reasoning: {thought}\n\n"

        response += f"<tool_call>\n{action_json_str}\n</tool_call>"
        return response

    def gen_next_round(self, line, state, previous_model_response=None, hindsight=False):
        """
        Build a single-turn prompt for step si.
        History of prior actions is flattened as text in the user message.
        Uses app-specific prompt template matching the SFT training data.
        """
        if state is None:
            state = {
                '_si': 0,
                'history_strings': [],
                '_app': _detect_app(line),
            }
        else:
            state = copy.deepcopy(state)

        si = state['_si']
        if si >= len(line['steps']):
            return None

        step = line['steps'][si]

        # If there's a previous model response (RL rollout), extract action for history
        # Prefer reasoning text (matches SFT training data history format),
        # fall back to function-call format
        if previous_model_response and si > 0:
            try:
                parsed = self.parse_response(previous_model_response)
                think_text = parsed.get('think', '')
                if think_text:
                    # Use reasoning text as history (matches training data)
                    hist_str = f"Step {si}: {think_text}"
                elif parsed.get('action_content'):
                    hist_str = _format_action_for_history(parsed['action_content'], si)
                else:
                    hist_str = f"Step {si}: (unparseable action)"
            except Exception:
                hist_str = f"Step {si}: (unparseable action)"
            state['history_strings'].append(hist_str)
        elif si > 0 and not previous_model_response:
            # Use GT thought from previous step for history (matches training data)
            prev_step = line['steps'][si - 1]
            prev_thought = prev_step.get('thought', '') or prev_step.get('motivation', '')
            if prev_thought:
                hist_str = f"Step {si}: {prev_thought}"
            else:
                hist_str = _format_action_for_history(prev_step['action_content'], si)
            state['history_strings'].append(hist_str)

        # Build history text
        if state['history_strings']:
            history_text = "\n".join(state['history_strings'])
        else:
            history_text = ""

        # Build user message text using app-specific template
        user_text = _build_user_text(line, history_text, app=state['_app'])

        # Build image element
        try:
            image_ele = make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None))
        except Exception:
            print(step['screenshot'])
            raise

        # Single-turn messages: user message with image + text
        messages = [{
            'role': 'user',
            'content': [
                image_ele,
                {"text": user_text},
            ]
        }]

        # Build messages_with_response (for GT training data)
        messages_with_response = copy.deepcopy(messages)
        model_response = ''
        if step.get('action_content', None):
            model_response = self.format_response(step, image_ele, add_thought=True)
        messages_with_response.append({
            'role': 'assistant',
            'content': [{"text": model_response}]
        })

        # Store thought for reward
        if 'thought' in step and step['thought'] != "":
            state['thought'] = step['thought']
        elif 'motivation' in step and step['motivation'] != "":
            state['thought'] = step['motivation']
        else:
            state['thought'] = ""

        # Call base class post-processing
        self._gen_round_post(line, state, image_ele, si, messages, messages_with_response)

        return state
