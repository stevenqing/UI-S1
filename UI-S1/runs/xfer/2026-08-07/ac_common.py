import ast
import functools
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
OFFICIAL_REPO = ROOT / "runs/androidcontrol-rft/2026-07-29/repo"
PROMPT_VARIABLES = {
    "UI-AGILE-7B": "ANDROID_CONTROL_DETAILED",
    "GUI-R1-7B": "GUI_R1_ANDROID_CONTROL",
    "UI-R1-E-3B": "UI_R1_ANDROID_CONTROL",
}
ACTION_MAP = {"navigate_back": "press_back", "input_text": "type"}
COORDINATE_ACTIONS = {"click", "long_press", "moveto", "doubleclick", "rightclick"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def image_sha256(sample):
    return hashlib.sha256(sample["image"]["bytes"]).hexdigest()


def source_sha256(sample):
    return canonical_hash({key: value for key, value in sample.items() if key != "image"})


def episode_key(instruction):
    return "ac_goal_" + hashlib.sha256(instruction.strip().encode()).hexdigest()[:20]


def load_prompt_templates():
    source = OFFICIAL_REPO / "eval/android_control/inference_android_control.py"
    tree = ast.parse(source.read_text())
    wanted = set(PROMPT_VARIABLES.values())
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                values[target.id] = ast.literal_eval(node.value)
    if set(values) != wanted:
        raise ValueError(f"missing official AC prompts: {sorted(wanted - set(values))}")
    return {model: values[variable] for model, variable in PROMPT_VARIABLES.items()}


def prompt_text(model_id, sample, templates=None):
    templates = load_prompt_templates() if templates is None else templates
    return templates[model_id].format(
        instruction=sample["instruction"], history=sample.get("history", "None")
    )


@functools.lru_cache(maxsize=1)
def load_official_parsers():
    source_dir = OFFICIAL_REPO / "eval/android_control"
    sys.path.insert(0, str(source_dir))
    from eval import extract_param_value_loosely, gui_r1_extract_param
    from utils import extract_action, extract_coordinates
    return extract_action, extract_coordinates, extract_param_value_loosely, gui_r1_extract_param


def parse_response(response, model_id, coordinate_scale):
    extract_action, extract_coordinates, extract_parameter, extract_gui_parameter = load_official_parsers()
    raw_action = extract_action(response)
    action = ACTION_MAP.get(raw_action, raw_action)
    coordinates, _, _ = extract_coordinates(response)
    position = None
    if coordinates is not None and len(coordinates) >= 2:
        position = [coordinates[0] * coordinate_scale[0], coordinates[1] * coordinate_scale[1]]
    parameter = extract_gui_parameter(response) if model_id == "GUI-R1-7B" else extract_parameter(response)
    return {
        "action": action,
        "value": parameter,
        "position": position,
        "parse_ok": action is not None,
    }


def load_paired_sample():
    manifest = [json.loads(line) for line in (RUN_DIR / "data/androidcontrol/subsample.jsonl").read_text().splitlines() if line.strip()]
    low = pq.read_table(RUN_DIR / "data/androidcontrol/androidcontrol_low_test.parquet").to_pylist()
    high = pq.read_table(RUN_DIR / "data/androidcontrol/androidcontrol_high_test.parquet").to_pylist()
    if len(manifest) != 2000 or len(low) != 7708 or len(high) != 7708:
        raise ValueError("AC paired data coverage mismatch")
    output = []
    for identity in manifest:
        low_row = low[identity["low_index"]]
        high_row = high[identity["high_index"]]
        if source_sha256(low_row) != identity["source_low_sha256"]:
            raise ValueError(f"AC Low source mismatch: {identity['id']}")
        if source_sha256(high_row) != identity["source_high_sha256"]:
            raise ValueError(f"AC High source mismatch: {identity['id']}")
        if low_row["gt_action"] != identity["gt_action"] or high_row["gt_action"] != identity["gt_action"]:
            raise ValueError(f"AC paired GT action mismatch: {identity['id']}")
        output.append({**identity, "low": low_row, "high": high_row, "episode_id": episode_key(high_row["instruction"])})
    return output


def fold_mapping(setting):
    folds = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())
    return folds["pools"][f"androidcontrol/{setting}"]["group_to_fold"]