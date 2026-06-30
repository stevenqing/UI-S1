#!/usr/bin/env python3
"""Build GUI-360 history-bearing SFT arms.

Arms:

- G / gt_history: teacher-forced multi-turn conversations. Prior turns contain
  expert actions; no model is required.
- O / own_history: semi-online own-thought rollout with expert-action patching.
  This requires a real Harness. The default harness is `_Unwired` and raises.

Invariants are enforced with assertions/raises:

- image markers across human turns equal len(images)
- arms use expert screenshots only; no post-divergence screens enter O
- own_history cannot run against the unwired harness
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui360_long_horizon.harness.correctness import coordinate_hit, function_match
from gui360_long_horizon.harness.model import VLLMClient
from gui360_long_horizon.harness.rollout import Harness, UnwiredHarness, VLLMHarness
from scripts.reproduce_gui360_fullparam_sft import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BALANCED_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_DS_CONFIG,
    DEFAULT_IMAGE_DIR,
    count_rows,
    image_bytes_from_cell,
    load_rows,
    safe_rel_image_path,
    write_dataset_info,
    write_image,
    write_train_yaml,
)


@dataclass(frozen=True)
class ArmBuildResult:
    arm: str
    examples_written: int
    episodes_read: int
    steps_seen: int
    mismatches: int
    skipped_episodes: int
    truncated_episodes: int
    output_json: str
    config_path: str


def _assert_image_count(conversations: Sequence[Dict[str, str]], images: Sequence[str]) -> None:
    markers = sum(str(turn.get("value", "")).count("<image>") for turn in conversations if turn.get("from") == "human")
    if markers != len(images):
        raise ValueError(f"image-count invariant failed: <image> markers={markers}, len(images)={len(images)}")


def _assert_expert_screenshot(step: Dict[str, Any]) -> None:
    screenshot = str(step.get("screenshot") or "").replace("\\", "/")
    if screenshot and "/fail/" in screenshot:
        raise ValueError(f"offline-wall invariant failed: own_history saw fail/post-divergence screenshot {screenshot}")


def _load_episode_steps(row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    try:
        steps = json.loads(row.get("steps") or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError("bad steps JSON") from exc
    screenshots_value = row.get("screenshots")
    screenshots = list(screenshots_value) if screenshots_value is not None else []
    return steps, screenshots


def _strip_inline_history(human: str) -> str:
    """Remove original single-step text history so multi-turn prior turns carry history."""

    text = str(human or "")
    pattern = re.compile(r"(The history of actions(?: and observations)? (?:is|are):)(.*?)(\n\s*The actions supported are:)", flags=re.IGNORECASE | re.DOTALL)

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)} None{match.group(3)}"

    stripped, count = pattern.subn(repl, text, count=1)
    if count:
        return stripped
    return text


def _extract_instruction(human: str) -> str:
    text = str(human or "")
    match = re.search(r"The instruction is:\s*(.*?)(?:\n\s*The history of actions|\n\s*The actions supported are:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    cleaned = text.replace("<image>", " ")
    return re.sub(r"\s+", " ", cleaned).strip()[:512]


def _compact_history_arm_human(human: str) -> str:
    instruction = _extract_instruction(human)
    return (
        "<image>\n"
        "You are a GUI action agent. Use the current screenshot, the user instruction, and previous conversation turns to choose the next action.\n\n"
        f"Instruction: {instruction}\n\n"
        "History: previous turns in this conversation. If there are no previous turns, history is None.\n\n"
        "Supported actions: click(coordinate=[x,y]), type(text=...), swipe(start_coordinate=[x,y], end_coordinate=[x,y]), press(key=...), wait().\n"
        "Output exactly one JSON action inside <tool_call></tool_call>."
    )


def _conversation_pair(step: Dict[str, Any], assistant_value: Optional[str] = None) -> List[Dict[str, str]]:
    human = _compact_history_arm_human(str(step.get("conversation_human") or "")).strip()
    assistant = str(assistant_value if assistant_value is not None else step.get("conversation_gpt") or "").strip()
    if not human or not assistant:
        raise ValueError("missing conversation_human/conversation_gpt")
    return [{"from": "human", "value": human}, {"from": "gpt", "value": assistant}]


def _expert_action_for_match(step: Dict[str, Any]) -> Dict[str, Any]:
    action = dict(step.get("action") or {})
    if "function" not in action and "action" in action:
        action["function"] = action.get("action")
    return action


def _gt_rect(step: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bbox = step.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        left, top, right, bottom = map(float, bbox[:4])
        return min(left, right), min(top, bottom), max(left, right), max(top, bottom)
    return None


def _matches_expert(pred_action: Dict[str, Any], step: Dict[str, Any]) -> bool:
    expert = _expert_action_for_match(step)
    rect = _gt_rect(step)
    if not function_match(pred_action, str(expert.get("function") or expert.get("action") or "")):
        return False
    if rect is None:
        return True
    coord = pred_action.get("coordinate") or pred_action.get("xy")
    if coord is None:
        return False
    return coordinate_hit((float(coord[0]), float(coord[1])), rect)


def _patched_assistant_value(prediction: HarnessPrediction, expert_action_text: str) -> str:
    thought = prediction.thought.strip()
    if thought:
        return thought + "\n" + expert_action_text.strip()
    return expert_action_text.strip()


def _write_step_image(
    *,
    split: str,
    episode_id: Any,
    step_idx: Any,
    step_pos: int,
    step: Dict[str, Any],
    screenshots: Sequence[Any],
    image_root: Path,
    require_images: bool,
) -> str:
    rel_image = safe_rel_image_path(split, episode_id, step_idx)
    img_cell = screenshots[step_pos] if step_pos < len(screenshots) else None
    path = write_image(
        image_root,
        rel_image,
        image_bytes_from_cell(img_cell),
        str(step.get("screenshot") or ""),
        require_image=require_images,
    )
    if not path:
        raise FileNotFoundError(f"could not resolve image for episode={episode_id} step={step_idx}")
    return path


def build_gt_history_example(
    row: Dict[str, Any],
    *,
    split: str,
    image_root: Path,
    require_images: bool,
) -> Optional[Dict[str, Any]]:
    episode_id = row.get("episode_id")
    steps, screenshots = _load_episode_steps(row)
    conversations: List[Dict[str, str]] = []
    images: List[str] = []
    for step_pos, step in enumerate(steps):
        _assert_expert_screenshot(step)
        step_idx = step.get("step_idx", step_pos)
        pair = _conversation_pair(step)
        image_path = _write_step_image(split=split, episode_id=episode_id, step_idx=step_idx, step_pos=step_pos, step=step, screenshots=screenshots, image_root=image_root, require_images=require_images)
        conversations.extend(pair)
        images.append(image_path)
    if not conversations:
        return None
    _assert_image_count(conversations, images)
    return {"conversations": conversations, "images": images}


def build_own_history_example(
    row: Dict[str, Any],
    *,
    split: str,
    image_root: Path,
    require_images: bool,
    harness: Harness,
    patch_budget: int,
) -> Tuple[Optional[Dict[str, Any]], int, bool]:
    episode_id = row.get("episode_id")
    steps, screenshots = _load_episode_steps(row)
    conversations: List[Dict[str, str]] = []
    images: List[str] = []
    mismatches = 0
    truncated = False
    for step_pos, step in enumerate(steps):
        _assert_expert_screenshot(step)
        step_idx = step.get("step_idx", step_pos)
        image_path = _write_step_image(split=split, episode_id=episode_id, step_idx=step_idx, step_pos=step_pos, step=step, screenshots=screenshots, image_root=image_root, require_images=require_images)
        pair = _conversation_pair(step)
        prediction = harness.predict(conversations, images + [image_path], step)
        if not _matches_expert(prediction.action, step):
            mismatches += 1
            if mismatches > patch_budget:
                truncated = True
                break
        conversations.append(pair[0])
        conversations.append({"from": "gpt", "value": _patched_assistant_value(prediction, pair[1]["value"])})
        images.append(image_path)
    if not conversations:
        return None, mismatches, truncated
    _assert_image_count(conversations, images)
    return {"conversations": conversations, "images": images}, mismatches, truncated


def build_arm_data(
    *,
    arm: str,
    split: str,
    balanced_data_dir: Path,
    output_json: Path,
    image_root: Path,
    max_episodes: int,
    start_episode: int,
    require_images: bool,
    patch_budget: int,
    harness: Harness,
    allow_skip_bad_episodes: bool = False,
) -> Dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    examples: List[Dict[str, Any]] = []
    episodes = 0
    rows_seen = 0
    steps_seen = 0
    mismatches = 0
    skipped = 0
    truncated = 0
    for row_index, row in enumerate(load_rows(balanced_data_dir, split, max_episodes=-1)):
        if row_index < start_episode:
            continue
        if max_episodes > 0 and episodes >= max_episodes:
            break
        rows_seen += 1
        episodes += 1
        try:
            steps, _ = _load_episode_steps(row)
            steps_seen += len(steps)
            if arm == "gt_history":
                example = build_gt_history_example(row, split=split, image_root=image_root, require_images=require_images)
                if example:
                    examples.append(example)
            elif arm == "own_history":
                example, mm, trunc = build_own_history_example(row, split=split, image_root=image_root, require_images=require_images, harness=harness, patch_budget=patch_budget)
                mismatches += mm
                truncated += int(trunc)
                if example:
                    examples.append(example)
            else:
                raise ValueError(f"unknown arm: {arm}")
        except Exception:
            skipped += 1
            if not allow_skip_bad_episodes:
                raise
    output_json.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    return {"arm": arm, "split": split, "start_episode": start_episode, "rows_seen": rows_seen, "episodes_read": episodes, "steps_seen": steps_seen, "examples_written": len(examples), "mismatches": mismatches, "skipped_episodes": skipped, "truncated_episodes": truncated, "output_json": str(output_json)}


def write_arm_yaml(config_path: Path, *, base_model: str, dataset_dir: str, output_dir: str, ds_config: str, dataset: str, eval_dataset: str, run_name: str, image_max_pixels: int, cutoff_len: int, save_strategy: str, eval_strategy: str) -> Path:
    return write_train_yaml(
        path=config_path,
        base_model=base_model,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        ds_config=ds_config,
        train_dataset=dataset,
        val_dataset=eval_dataset,
        image_max_pixels=image_max_pixels,
        cutoff_len=cutoff_len,
        epochs=4.0,
        learning_rate=1.0e-5,
        gradient_accumulation_steps=16,
        save_strategy=save_strategy,
        save_steps=50,
        eval_strategy=eval_strategy,
        eval_steps=50,
        report_to="none",
        run_name=run_name,
    )


def _resolve_train_val_windows(args: argparse.Namespace, balanced_dir: Path) -> Tuple[int, int, int]:
    train_start = max(args.train_start_episode, 0)
    train_max = args.max_train_episodes
    val_start = args.val_start_episode
    if val_start >= 0:
        return train_start, train_max, val_start
    if args.val_source_split == "train" and args.max_val_episodes > 0:
        total_train = count_rows(balanced_dir, "train")
        val_start = max(0, total_train - args.max_val_episodes)
        if train_max < 0 and train_start == 0:
            train_max = val_start
        return train_start, train_max, val_start
    return train_start, train_max, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GUI-360 history-utilization SFT arms")
    parser.add_argument("--arm", choices=["gt_history", "own_history"], required=True)
    parser.add_argument("--balanced-data-dir", default="datasets/gui360-balanced/data")
    parser.add_argument("--data-dir", default="train_GUI_360/llamafactory/data")
    parser.add_argument("--image-dir", default="train_GUI_360/llamafactory/data/gui360_history_arm_images")
    parser.add_argument("--base-model", default="checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--ds-config", default="train_GUI_360/llamafactory/ds_z3_config.json")
    parser.add_argument("--max-train-episodes", type=int, default=-1)
    parser.add_argument("--max-val-episodes", type=int, default=32)
    parser.add_argument("--train-start-episode", type=int, default=0)
    parser.add_argument("--val-start-episode", type=int, default=-1)
    parser.add_argument("--val-source-split", choices=["train", "test"], default="train", help="GUI-360 balanced test has action labels but no conversation text; train tail is the default SFT eval source.")
    parser.add_argument("--patch-budget", type=int, default=3)
    parser.add_argument("--image-max-pixels", type=int, default=200704, help="Capstone default lowers per-image tokens so multi-turn history fits the matched cutoff.")
    parser.add_argument("--cutoff-len", type=int, default=16384)
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch", "best"])
    parser.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--harness-base-url", default="", help="OpenAI-compatible endpoint for own_history data construction")
    parser.add_argument("--harness-model", default="", help="Model name served at --harness-base-url; defaults to --base-model")
    parser.add_argument("--harness-timeout", type=float, default=600.0)
    parser.add_argument("--harness-max-tokens", type=int, default=256)
    parser.add_argument("--harness-temperature", type=float, default=0.0)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_root = Path(args.image_dir) / args.arm
    balanced_dir = Path(args.balanced_data_dir)
    train_start, train_max, val_start = _resolve_train_val_windows(args, balanced_dir)
    train_name = f"gui360_{args.arm}_train"
    val_name = f"gui360_{args.arm}_val"
    train_json = f"{train_name}.json"
    val_json = f"{val_name}.json"
    output_dir = f"train_GUI_360/llamafactory/output/gui360_{args.arm}_full_sft"
    config_path = Path(f"train_GUI_360/llamafactory/qwen25vl_gui360_{args.arm}_full_sft.yaml")
    if args.arm == "own_history" and args.harness_base_url:
        harness = VLLMHarness(
            VLLMClient(args.harness_base_url, args.harness_model or args.base_model, timeout=args.harness_timeout),
            max_tokens=args.harness_max_tokens,
            temperature=args.harness_temperature,
        )
    else:
        harness = UnwiredHarness()
    summary: Dict[str, Any] = {"arm": args.arm, "train_dataset": train_name, "val_dataset": val_name, "val_source_split": args.val_source_split, "train_start_episode": train_start, "train_max_episodes": train_max, "val_start_episode": val_start}

    if args.prepare_data:
        summary["train"] = build_arm_data(arm=args.arm, split="train", balanced_data_dir=balanced_dir, output_json=data_dir / train_json, image_root=image_root, max_episodes=train_max, start_episode=train_start, require_images=args.require_images, patch_budget=args.patch_budget, harness=harness)
        summary["val"] = build_arm_data(arm=args.arm, split=args.val_source_split, balanced_data_dir=balanced_dir, output_json=data_dir / val_json, image_root=image_root, max_episodes=args.max_val_episodes, start_episode=val_start, require_images=args.require_images, patch_budget=args.patch_budget, harness=harness, allow_skip_bad_episodes=True)
        if summary["val"]["examples_written"] == 0:
            raise RuntimeError(f"validation conversion produced zero examples from split={args.val_source_split}; use --val-source-split train for history-SFT eval")
        summary["dataset_info"] = str(write_dataset_info(data_dir, train_name, val_name, train_json, val_json))
    if args.write_config:
        summary["config"] = str(write_arm_yaml(config_path, base_model=args.base_model, dataset_dir=args.data_dir, output_dir=output_dir, ds_config=args.ds_config, dataset=train_name, eval_dataset=val_name, run_name=f"gui360_{args.arm}_full_sft", image_max_pixels=args.image_max_pixels, cutoff_len=args.cutoff_len, save_strategy=args.save_strategy, eval_strategy=args.eval_strategy))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
