#!/usr/bin/env python3
"""
Task 3.1: Prepare data for VLM eigenfunction training.

Maps state_hash → screenshot path + a11y tree text, then produces
per-app state_manifest.json and transition_pairs.json.

Inputs:
    outputs/transitions/gui360_full/transitions.jsonl
    outputs/fnet/gui360/{app}/f_values.npz
    datasets/GUI-360/train/image/{app}/{cat}/success/
    datasets/GUI-360/train/data/{app}/{cat}/success/

Outputs per-app:
    outputs/vlm_fnet/data/{app}/state_manifest.json
        [{hash, f_value, screenshot_path, a11y_text}, ...]
    outputs/vlm_fnet/data/{app}/transition_pairs.json
        [{src_hash, dst_hash}, ...]

Usage:
    python scripts/prepare_vlm_eigenfunction_data.py
    python scripts/prepare_vlm_eigenfunction_data.py --app excel
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
DATASET_DIR = PROJECT_ROOT / "datasets" / "GUI-360" / "train"
TRANSITIONS_PATH = PROJECT_ROOT / "outputs" / "transitions" / "gui360_full" / "transitions.jsonl"
FNET_DIR = PROJECT_ROOT / "outputs" / "fnet" / "gui360"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vlm_fnet" / "data"

APPS = ["excel", "word", "ppt"]
CATEGORIES = ["in_app", "online", "search"]


def compress_ui_tree(ui_tree: dict) -> str:
    """Compress a UI tree dict into a short text summary.

    Output format:
        Window: Excel | Tab: Data,Insert,... | Dialog: Format Cells | Controls: Button:3,Edit:1,...

    Truncated to ~200 chars.
    """
    if not isinstance(ui_tree, dict):
        return ""

    parts = []

    # Window name
    root_name = ui_tree.get("name", "")
    if root_name:
        # Extract app name from window title (e.g., "xxx - Excel" → "Excel")
        for app in ["Excel", "Word", "PowerPoint"]:
            if app in root_name:
                parts.append(f"Window: {app}")
                break
        else:
            # Use first 30 chars of window name
            parts.append(f"Window: {root_name[:30]}")

    # Scan level-1 children for tabs, dialogs, and control counts
    children = ui_tree.get("children", [])
    tabs = []
    dialogs = []
    control_counts: dict[str, int] = defaultdict(int)

    def _count_controls(node: dict, depth: int = 0):
        """Recursively count controls by type."""
        ct = node.get("control_type", "")
        name = node.get("name", "")

        if ct == "TabItem" and name:
            tabs.append(name)
        elif ct == "Window" and depth == 1 and name:
            dialogs.append(name)

        if ct and ct not in ("Window", "Pane", "Group"):
            control_counts[ct] += 1

        for child in node.get("children", []):
            _count_controls(child, depth + 1)

    _count_controls(ui_tree)

    # Tab signature
    if tabs:
        tab_str = ",".join(tabs[:10])
        parts.append(f"Tab: {tab_str}")

    # Dialog state
    if dialogs:
        dialog_str = ",".join(dialogs[:3])
        parts.append(f"Dialog: {dialog_str}")

    # Top control types by count
    if control_counts:
        sorted_controls = sorted(control_counts.items(), key=lambda x: -x[1])[:6]
        ctrl_str = ",".join(f"{ct}:{n}" for ct, n in sorted_controls)
        parts.append(f"Controls: {ctrl_str}")

    result = " | ".join(parts)
    if len(result) > 200:
        result = result[:197] + "..."
    return result


def build_hash_to_location_map(transitions_path: Path) -> dict[str, list[tuple[str, int]]]:
    """Build mapping: state_hash → [(execution_id, step_id), ...].

    Uses transitions.jsonl which records (state_hash, execution_id, step_id)
    for every transition.
    """
    logger.info(f"Reading transitions from {transitions_path}")
    hash_to_locs: dict[str, list[tuple[str, int]]] = defaultdict(list)

    with open(transitions_path) as f:
        for line in f:
            rec = json.loads(line)
            h = rec["state_hash"]
            exec_id = rec["execution_id"]
            step_id = rec["step_id"]
            hash_to_locs[h].append((exec_id, step_id))
            # Also add next_state_hash from the last step
            nh = rec["next_state_hash"]
            next_step = step_id + 1
            hash_to_locs[nh].append((exec_id, next_step))

    logger.info(f"Built location map for {len(hash_to_locs)} unique hashes")
    return dict(hash_to_locs)


def _list_screenshots_sorted(img_dir: Path) -> list[Path]:
    """List action_step*.png files sorted by step number."""
    import re
    files = []
    if not img_dir.is_dir():
        return files
    for p in img_dir.iterdir():
        m = re.match(r"action_step(\d+)\.png$", p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def find_screenshot_path(
    exec_id: str, step_id: int, app: str, dataset_dir: Path
) -> Path | None:
    """Find the screenshot file for a given (execution_id, step_id).

    GUI-360 screenshot filenames can have numbering gaps (e.g., 1,2,3,5,7)
    while data JSONL step_ids are consecutive (1,2,3,4,5). We match by
    index: step_id K → the K-th screenshot file (1-indexed, sorted by number).
    """
    for cat in CATEGORIES:
        img_dir = dataset_dir / "image" / app / cat / "success" / exec_id
        # Fast path: try direct match first (covers ~75% of cases)
        img_path = img_dir / f"action_step{step_id}.png"
        if img_path.exists():
            return img_path
        # Slow path: index-based lookup for directories with numbering gaps
        sorted_files = _list_screenshots_sorted(img_dir)
        idx = step_id - 1  # 1-indexed → 0-indexed
        if 0 <= idx < len(sorted_files):
            return sorted_files[idx]
    return None


def find_data_path(exec_id: str, app: str, dataset_dir: Path) -> Path | None:
    """Find the JSONL data file for a given execution_id."""
    for cat in CATEGORIES:
        data_path = dataset_dir / "data" / app / cat / "success" / f"{exec_id}.jsonl"
        if data_path.exists():
            return data_path
    return None


def load_step_data(data_path: Path, step_id: int) -> dict | None:
    """Load a specific step from a JSONL file."""
    with open(data_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("step_id") == step_id:
                return rec
    return None


def extract_a11y_text(step_data: dict) -> str:
    """Extract and compress a11y text from step data."""
    step = step_data.get("step", {})
    if isinstance(step, str):
        try:
            step = json.loads(step)
        except json.JSONDecodeError:
            return ""
    ui_tree = step.get("ui_tree", {})
    if not ui_tree:
        return ""
    return compress_ui_tree(ui_tree)


def prepare_app_data(
    app: str,
    hash_to_locs: dict[str, list[tuple[str, int]]],
    fnet_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    transitions_path: Path,
    max_states: int | None = None,
):
    """Prepare state_manifest.json and transition_pairs.json for one app."""
    app_output = output_dir / app
    app_output.mkdir(parents=True, exist_ok=True)

    # Load MLP f-values for this app (used for comparison/validation)
    fval_path = fnet_dir / app / "f_values.npz"
    if not fval_path.exists():
        logger.warning(f"f_values.npz not found for {app} at {fval_path}, skipping f-values")
        app_hashes_set = set()
        f_value_map = {}
    else:
        fv_data = np.load(fval_path, allow_pickle=True)
        app_hashes = fv_data["hashes"]
        app_f_values = fv_data["f_values"]
        app_hashes_set = set(str(h) for h in app_hashes)
        f_value_map = {str(h): float(fv) for h, fv in zip(app_hashes, app_f_values)}
        logger.info(f"{app}: {len(app_hashes_set)} states from f_values.npz")

    # Build manifest: for each state hash, find a screenshot and a11y text
    manifest = []
    missing_screenshot = 0
    missing_data = 0

    # Filter hashes that belong to this app
    app_prefix = app + "_"
    relevant_hashes = set()
    for h, locs in hash_to_locs.items():
        for exec_id, _ in locs:
            if exec_id.startswith(app_prefix):
                relevant_hashes.add(h)
                break

    # If we have f_values, intersect
    if app_hashes_set:
        relevant_hashes = relevant_hashes & app_hashes_set

    logger.info(f"{app}: {len(relevant_hashes)} relevant hashes to process")

    if max_states and len(relevant_hashes) > max_states:
        relevant_hashes = set(list(relevant_hashes)[:max_states])
        logger.info(f"{app}: Limited to {max_states} states for testing")

    for h in sorted(relevant_hashes):
        locs = hash_to_locs.get(h, [])
        if not locs:
            continue

        # Find a valid screenshot from any location
        screenshot_path = None
        a11y_text = ""

        for exec_id, step_id in locs:
            if not exec_id.startswith(app_prefix):
                continue

            # Try to find screenshot
            sp = find_screenshot_path(exec_id, step_id, app, dataset_dir)
            if sp is None:
                continue

            screenshot_path = sp

            # Try to find a11y text
            data_path = find_data_path(exec_id, app, dataset_dir)
            if data_path:
                step_data = load_step_data(data_path, step_id)
                if step_data:
                    a11y_text = extract_a11y_text(step_data)

            # Found both, done
            if screenshot_path and a11y_text:
                break

        if screenshot_path is None:
            missing_screenshot += 1
            continue

        if not a11y_text:
            missing_data += 1

        entry = {
            "hash": h,
            "f_value": f_value_map.get(h, 0.0),
            "screenshot_path": str(screenshot_path),
            "a11y_text": a11y_text,
        }
        manifest.append(entry)

    logger.info(
        f"{app}: Built manifest with {len(manifest)} states "
        f"(missing screenshot: {missing_screenshot}, missing a11y: {missing_data})"
    )

    # Save manifest
    manifest_path = app_output / "state_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Build transition pairs (only for hashes in manifest)
    manifest_hashes = {e["hash"] for e in manifest}
    transitions = []

    with open(transitions_path) as f:
        for line in f:
            rec = json.loads(line)
            src_h = rec["state_hash"]
            dst_h = rec["next_state_hash"]
            if src_h in manifest_hashes and dst_h in manifest_hashes:
                transitions.append({"src_hash": src_h, "dst_hash": dst_h})

    logger.info(f"{app}: {len(transitions)} transition pairs")

    pairs_path = app_output / "transition_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(transitions, f)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  {app.upper()} — Data Preparation Summary")
    print(f"{'='*50}")
    print(f"  States in manifest: {len(manifest)}")
    print(f"  Transition pairs: {len(transitions)}")
    print(f"  Missing screenshots: {missing_screenshot}")
    print(f"  Missing a11y data: {missing_data}")
    if manifest:
        with_a11y = sum(1 for e in manifest if e["a11y_text"])
        print(f"  States with a11y text: {with_a11y} ({100*with_a11y/len(manifest):.1f}%)")
        avg_a11y_len = np.mean([len(e["a11y_text"]) for e in manifest if e["a11y_text"]])
        print(f"  Avg a11y text length: {avg_a11y_len:.0f} chars")
    print(f"  Output: {app_output}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for VLM eigenfunction training (Task 3.1)"
    )
    parser.add_argument(
        "--app", type=str, choices=APPS + ["all"], default="all",
        help="Prepare data for a single app or all apps (default: all)",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DATASET_DIR,
        help="GUI-360 dataset directory",
    )
    parser.add_argument(
        "--transitions-path", type=Path, default=TRANSITIONS_PATH,
        help="Path to transitions.jsonl",
    )
    parser.add_argument(
        "--fnet-dir", type=Path, default=FNET_DIR,
        help="Directory containing MLP f-values (outputs/fnet/gui360)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--max-states", type=int, default=None,
        help="Limit number of states per app (for testing)",
    )
    args = parser.parse_args()

    if not args.transitions_path.exists():
        logger.error(f"Transitions file not found: {args.transitions_path}")
        sys.exit(1)

    # Build hash → location map (shared across all apps)
    hash_to_locs = build_hash_to_location_map(args.transitions_path)

    apps_to_process = APPS if args.app == "all" else [args.app]

    for app in apps_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Preparing data for: {app}")
        logger.info(f"{'='*60}")

        prepare_app_data(
            app=app,
            hash_to_locs=hash_to_locs,
            fnet_dir=args.fnet_dir,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            transitions_path=args.transitions_path,
            max_states=args.max_states,
        )

    logger.info("\nData preparation complete!")


if __name__ == "__main__":
    main()
