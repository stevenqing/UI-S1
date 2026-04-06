"""Convert GUI-Odyssey annotations to AC-compatible JSONL for evaluation.

Input:  GUI-Odyssey annotations/*.json + splits/*.json
Output: gui_odyssey_{split}_test.jsonl  (one episode per line)

Coordinate system: GUI-Odyssey uses [0,1000] normalized coordinates.
We store them as-is and handle normalization in the action matching function.
"""

import argparse
import json
import os
import sys


def get_scroll_direction(coord1, coord2):
    """Derive scroll direction from start/end coordinates in [0,1000] space."""
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'


def convert_action(step):
    """Convert GUI-Odyssey step action/info to AC-compatible action_content.

    Returns:
        action_content dict compatible with the AC evaluation pipeline.
    """
    action = step['action']
    info = step['info']

    # Special key presses (CLICK on KEY_*)
    if action == 'CLICK' and isinstance(info, str) and info.startswith('KEY_'):
        key_map = {
            'KEY_HOME': 'Home',
            'KEY_BACK': 'Back',
            'KEY_APPSELECT': 'Menu',
        }
        button = key_map.get(info, info.replace('KEY_', ''))
        return {'action': 'system_button', 'button': button}

    if action == 'CLICK':
        # info is [[x,y],[x,y]] — use first point as click coordinate
        coord = info[0] if isinstance(info, list) and len(info) > 0 else [0, 0]
        return {'action': 'click', 'coordinate': list(coord)}

    if action == 'LONG_PRESS':
        coord = info[0] if isinstance(info, list) and len(info) > 0 else [0, 0]
        return {'action': 'long_press', 'coordinate': list(coord)}

    if action == 'SCROLL':
        # info is [[x1,y1],[x2,y2]]
        coord1 = info[0] if isinstance(info, list) and len(info) > 0 else [0, 0]
        coord2 = info[1] if isinstance(info, list) and len(info) > 1 else [0, 0]
        return {
            'action': 'swipe',
            'coordinate': list(coord1),
            'coordinate2': list(coord2),
        }

    if action == 'TEXT':
        # info is the text string
        return {'action': 'type', 'text': str(info)}

    if action == 'COMPLETE':
        return {'action': 'terminate', 'status': 'success'}

    if action == 'INCOMPLETE':
        return {'action': 'terminate', 'status': 'failure'}

    raise ValueError(f"Unknown action: {action}, info: {info}")


def convert_sam2_bbox(sam2_bbox):
    """Convert sam2_bbox [x1,y1,x2,y2] to candidate_bbox [[x1,y1,x2,y2]] format.

    Both are in [0,1000] normalized space.
    Returns empty list if sam2_bbox is empty or None.
    """
    if not sam2_bbox:
        return []
    return [list(sam2_bbox)]


def convert_episode(episode_id, annotation, screenshot_root):
    """Convert a single GUI-Odyssey episode to AC-compatible format.

    Args:
        episode_id: Episode identifier string.
        annotation: Dict loaded from annotations/{episode_id}.json.
        screenshot_root: Absolute path to screenshots directory.

    Returns:
        Dict in AC-compatible format with keys:
        episode_id, goal, category, device_name, width, height, steps[]
    """
    device_info = annotation['device_info']
    task_info = annotation['task_info']

    converted_steps = []
    for step in annotation['steps']:
        # Build screenshot absolute path
        screenshot_path = os.path.join(screenshot_root, step['screenshot'])

        action_content = convert_action(step)

        # Build check_options: action_content + candidate_bbox from sam2_bbox
        check_options = dict(action_content)
        check_options['candidate_bbox'] = convert_sam2_bbox(step.get('sam2_bbox', []))

        converted_step = {
            'screenshot': screenshot_path,
            'action_content': action_content,
            'check_options': check_options,
            'step_instruction': step.get('low_level_instruction', ''),
            'thought': step.get('intention', ''),
        }
        converted_steps.append(converted_step)

    return {
        'episode_id': episode_id,
        'goal': task_info['instruction'],
        'category': task_info['category'],
        'device_name': device_info['device_name'],
        'width': device_info['w'],
        'height': device_info['h'],
        'steps': converted_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert GUI-Odyssey to AC-compatible JSONL")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory of GUI-Odyssey dataset')
    parser.add_argument('--split', type=str, default='random_split',
                        choices=['random_split', 'app_split', 'device_split', 'task_split'],
                        help='Which split to convert')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_dir)')
    parser.add_argument('--subset', type=str, default='test',
                        choices=['train', 'test'],
                        help='Which subset to convert')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load split
    split_path = os.path.join(data_dir, 'splits', f'{args.split}.json')
    with open(split_path) as f:
        split_data = json.load(f)
    episode_ids = split_data[args.subset]
    print(f"Split: {args.split}/{args.subset} — {len(episode_ids)} episodes")

    # Screenshot root — use absolute path so eval works from any cwd
    screenshot_root = os.path.abspath(os.path.join(data_dir, 'data', 'screenshots', 'screenshots'))
    if not os.path.isdir(screenshot_root):
        screenshot_root = os.path.abspath(os.path.join(data_dir, 'data', 'screenshots'))
    print(f"Screenshot root: {screenshot_root}")

    annotation_dir = os.path.join(data_dir, 'annotations')

    output_path = os.path.join(output_dir, f'gui_odyssey_{args.split}_{args.subset}.jsonl')

    converted = 0
    skipped = 0
    with open(output_path, 'w') as out_f:
        for eid in episode_ids:
            # Split IDs may include '.json' extension — strip it for annotation lookup
            eid_clean = eid.replace('.json', '') if eid.endswith('.json') else eid
            ann_path = os.path.join(annotation_dir, f'{eid_clean}.json')
            if not os.path.exists(ann_path):
                print(f"Warning: annotation not found: {ann_path}")
                skipped += 1
                continue

            with open(ann_path) as f:
                annotation = json.load(f)

            try:
                episode = convert_episode(eid_clean, annotation, screenshot_root)
                out_f.write(json.dumps(episode, ensure_ascii=False) + '\n')
                converted += 1
            except Exception as e:
                print(f"Error converting {eid}: {e}")
                skipped += 1

    print(f"Done. Converted: {converted}, Skipped: {skipped}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
