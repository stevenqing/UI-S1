# Copyright 2024 UI-S1 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
State representation utilities for graph-based option discovery on GUI-360.

Provides two complementary state representations:
- Hash-based state IDs (lightweight, for graph construction)
- Dense state embeddings (for neural eigenfunction training)

These are used in the spectral graph analysis pipeline (Tasks 2-3) to build
transition graphs and train the Laplacian eigenfunction approximator (f_net).

Reference: Jinnai et al. (2020), "Exploration in RL with Deep Covering Options", ICLR 2020.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# GUI-360 application domains
APP_DOMAINS = ("excel", "word", "ppt")
APP_DOMAIN_TO_IDX = {app: i for i, app in enumerate(APP_DOMAINS)}

# Canonical ribbon tab names across Office apps.
# Tabs are ordered by typical appearance; less common tabs are grouped into "other".
CANONICAL_TABS = (
    "Home", "Insert", "Draw", "Design", "Layout", "Page Layout",
    "References", "Mailings", "Review", "View", "Help", "Developer",
    "Transitions", "Animations", "Slide Show", "Record",
    "Data", "Formulas", "Automate",
)
CANONICAL_TAB_TO_IDX = {t: i for i, t in enumerate(CANONICAL_TABS)}
TAB_OTHER_IDX = len(CANONICAL_TABS)  # catch-all for unknown tabs

# Canonical UI control types from UIA (sorted by frequency in GUI-360).
CONTROL_TYPES = (
    "DataItem", "Button", "TabItem", "MenuItem", "Edit", "ListItem",
    "ScrollBar", "ComboBox", "Hyperlink", "CheckBox", "TreeItem",
    "Document", "RadioButton", "Spinner", "Image", "Text", "Slider",
)
CONTROL_TYPE_TO_IDX = {t: i for i, t in enumerate(CONTROL_TYPES)}
CONTROL_TYPE_OTHER_IDX = len(CONTROL_TYPES)
NUM_CONTROL_TYPE_BINS = len(CONTROL_TYPES) + 1  # +1 for "other"

# Bucketing thresholds for control counts (used in hash-based state ID).
COUNT_BUCKETS = (0, 5, 15, 50, 150, 500)


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class GUI360StepData:
    """Structured representation of a single GUI-360 trajectory step.

    This normalizes the raw JSONL step format into a clean interface
    for state extraction.
    """
    execution_id: str
    app_domain: str  # "excel" | "word" | "ppt"
    step_id: int
    total_steps: int
    request: str

    # Extracted from step.control_infos.application_windows_info
    window_title: str

    # Extracted from step.control_infos.merged_controls_info
    control_type_counts: dict[str, int]
    tab_names: list[str]
    total_controls: int

    # Extracted from step.ui_tree (level-1 children)
    dialog_names: list[str]  # names of Window-type children in ui_tree

    # Action info
    action_type: str  # "GUI" | "API"
    action_function: str  # "click", "type", "scroll", etc.
    status: str  # "CONTINUE" | "FINISH" | "OVERALL_FINISH"

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "GUI360StepData":
        """Parse a raw JSONL line (dict) into structured step data."""
        step = raw["step"]
        control_infos = step.get("control_infos", {})
        merged = control_infos.get("merged_controls_info", [])
        app_windows = control_infos.get("application_windows_info", {})
        ui_tree = step.get("ui_tree", {})
        action = step.get("action", {})

        # Count control types
        type_counts: dict[str, int] = {}
        tab_names: list[str] = []
        for ctrl in merged:
            ctype = ctrl.get("control_type", "Unknown")
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            if ctype == "TabItem":
                text = ctrl.get("control_text", "")
                if text:
                    tab_names.append(text)

        # Extract dialog names from ui_tree level-1 Window children
        dialog_names = []
        for child in ui_tree.get("children", []):
            if child.get("control_type") == "Window" and child.get("name"):
                dialog_names.append(child["name"])

        return cls(
            execution_id=raw.get("execution_id", ""),
            app_domain=raw.get("app_domain", ""),
            step_id=raw.get("step_id", 0),
            total_steps=raw.get("total_steps", 0),
            request=raw.get("request", ""),
            window_title=app_windows.get("control_text", ""),
            control_type_counts=type_counts,
            tab_names=tab_names,
            total_controls=len(merged),
            dialog_names=dialog_names,
            action_type=action.get("action_type", ""),
            action_function=action.get("function", ""),
            status=step.get("status", ""),
        )


@dataclass
class StateID:
    """Hash-based state identifier for graph construction.

    Captures the structural fingerprint of a UI state:
    - Which app (excel / word / ppt)
    - Which ribbon tabs are visible (proxy for current mode)
    - Whether a dialog is open and which one
    - The distribution of control types (bucketed)
    """
    app_domain: str
    active_tab_signature: str  # sorted, comma-joined tab names
    dialog_state: str  # "none" or dialog name
    control_fingerprint: str  # bucketed control type distribution
    hash_value: str = field(init=False)

    def __post_init__(self):
        key = f"{self.app_domain}|{self.active_tab_signature}|{self.dialog_state}|{self.control_fingerprint}"
        self.hash_value = hashlib.md5(key.encode()).hexdigest()

    def __hash__(self):
        return hash(self.hash_value)

    def __eq__(self, other):
        if not isinstance(other, StateID):
            return NotImplemented
        return self.hash_value == other.hash_value

    def __repr__(self):
        dialog_str = self.dialog_state if self.dialog_state != "none" else ""
        return f"State({self.app_domain}, tabs=[{self.active_tab_signature}], dialog={dialog_str})"

    def to_dict(self) -> dict:
        return {
            "app_domain": self.app_domain,
            "active_tab_signature": self.active_tab_signature,
            "dialog_state": self.dialog_state,
            "control_fingerprint": self.control_fingerprint,
            "hash": self.hash_value,
        }


# ============================================================================
# Core functions
# ============================================================================

def _bucket_count(count: int) -> int:
    """Map a count to a bucket index for coarser state hashing."""
    for i, threshold in enumerate(COUNT_BUCKETS):
        if count <= threshold:
            return i
    return len(COUNT_BUCKETS)


def _build_control_fingerprint(type_counts: dict[str, int]) -> str:
    """Build a bucketed control-type distribution string.

    Groups control counts into buckets to avoid over-splitting states
    due to minor count variations (e.g., 359 vs 360 DataItems).
    """
    parts = []
    for ctype in CONTROL_TYPES:
        cnt = type_counts.get(ctype, 0)
        bucket = _bucket_count(cnt)
        if bucket > 0:  # skip zero-count types
            parts.append(f"{ctype}:{bucket}")
    # Aggregate remaining types
    other_count = sum(
        cnt for t, cnt in type_counts.items() if t not in CONTROL_TYPE_TO_IDX
    )
    if other_count > 0:
        parts.append(f"Other:{_bucket_count(other_count)}")
    return ",".join(sorted(parts))


def _normalize_tab_signature(tab_names: list[str]) -> str:
    """Create a canonical tab signature from visible tab names.

    Filters out non-standard/plugin tabs and sorts canonically.
    """
    canonical = []
    for name in tab_names:
        if name in CANONICAL_TAB_TO_IDX:
            canonical.append(name)
    # Sort by canonical order
    canonical.sort(key=lambda t: CANONICAL_TAB_TO_IDX.get(t, 999))
    return ",".join(canonical)


def _extract_dialog_state(dialog_names: list[str]) -> str:
    """Extract the dialog state from ui_tree level-1 Window children.

    Returns the name of the first dialog, or "none" if no dialogs are open.
    Dialogs like "Format Cells", "Paragraph", "Remove Duplicates" are
    important state-discriminating features.
    """
    if not dialog_names:
        return "none"
    # Use the first (topmost) dialog name
    return dialog_names[0]


def extract_state_id(
    step_data: GUI360StepData,
    granularity: str = "fine",
) -> StateID:
    """Extract a hash-based state ID from a GUI-360 step.

    This is the lightweight state representation (Approach A from the
    Instruction doc, Section 5.2). It produces a deterministic hash
    that uniquely identifies the "structural mode" of the UI.

    Args:
        step_data: Parsed step data.
        granularity: Controls state abstraction level:
            - "fine": app + tabs + dialog + control_fingerprint (6,511 states)
            - "coarse": app + tabs + dialog only (higher reuse, ~600 states)
              Drops control_fingerprint to increase state merging.
              Better for graph visualization and connectivity analysis.

    Components:
    - app_domain: excel / word / ppt
    - active_tab_signature: which ribbon tabs are visible (sorted)
    - dialog_state: name of open dialog, or "none"
    - control_fingerprint: bucketed distribution of control types

    Args:
        step_data: Parsed step data from a GUI-360 trajectory.

    Returns:
        StateID with a hash_value suitable for graph node identification.
    """
    tab_sig = _normalize_tab_signature(step_data.tab_names)
    dialog = _extract_dialog_state(step_data.dialog_names)

    if granularity == "coarse":
        fingerprint = ""  # drop control fingerprint for higher state reuse
    else:
        fingerprint = _build_control_fingerprint(step_data.control_type_counts)

    return StateID(
        app_domain=step_data.app_domain,
        active_tab_signature=tab_sig,
        dialog_state=dialog,
        control_fingerprint=fingerprint,
    )


def extract_state_id_from_raw(
    raw_step: dict[str, Any],
    granularity: str = "fine",
) -> StateID:
    """Convenience: extract state ID directly from a raw JSONL dict."""
    step_data = GUI360StepData.from_raw(raw_step)
    return extract_state_id(step_data, granularity=granularity)


# ============================================================================
# Dense state embedding
# ============================================================================

def extract_state_embedding(step_data: GUI360StepData) -> np.ndarray:
    """Extract a dense vector representation of a GUI-360 UI state.

    The embedding concatenates:
    1. App domain one-hot (3 dims)
    2. Control type distribution (normalized, NUM_CONTROL_TYPE_BINS dims)
    3. Tab presence vector (len(CANONICAL_TABS) + 1 dims)
    4. Dialog indicator (1 dim: 0 or 1)
    5. Total controls count (1 dim, log-scaled)

    Total dimensionality: 3 + 18 + 20 + 1 + 1 = 43

    This is used as input to f_net (the eigenfunction approximator, Task 3).

    Args:
        step_data: Parsed step data from a GUI-360 trajectory.

    Returns:
        numpy array of shape (embedding_dim,).
    """
    parts = []

    # 1. App domain one-hot (3 dims)
    app_vec = np.zeros(len(APP_DOMAINS), dtype=np.float32)
    idx = APP_DOMAIN_TO_IDX.get(step_data.app_domain)
    if idx is not None:
        app_vec[idx] = 1.0
    parts.append(app_vec)

    # 2. Control type distribution (NUM_CONTROL_TYPE_BINS dims)
    ctrl_vec = np.zeros(NUM_CONTROL_TYPE_BINS, dtype=np.float32)
    total = max(step_data.total_controls, 1)  # avoid division by zero
    for ctype, count in step_data.control_type_counts.items():
        idx = CONTROL_TYPE_TO_IDX.get(ctype, CONTROL_TYPE_OTHER_IDX)
        ctrl_vec[idx] += count
    ctrl_vec /= total  # normalize to proportions
    parts.append(ctrl_vec)

    # 3. Tab presence vector (len(CANONICAL_TABS) + 1 dims)
    tab_vec = np.zeros(len(CANONICAL_TABS) + 1, dtype=np.float32)
    for tab_name in step_data.tab_names:
        idx = CANONICAL_TAB_TO_IDX.get(tab_name, TAB_OTHER_IDX)
        tab_vec[idx] = 1.0
    parts.append(tab_vec)

    # 4. Dialog indicator (1 dim)
    dialog_vec = np.array(
        [1.0 if step_data.dialog_names else 0.0], dtype=np.float32
    )
    parts.append(dialog_vec)

    # 5. Total controls count, log-scaled (1 dim)
    count_vec = np.array(
        [np.log1p(step_data.total_controls)], dtype=np.float32
    )
    parts.append(count_vec)

    return np.concatenate(parts)


def extract_state_embedding_from_raw(raw_step: dict[str, Any]) -> np.ndarray:
    """Convenience: extract state embedding directly from a raw JSONL dict."""
    step_data = GUI360StepData.from_raw(raw_step)
    return extract_state_embedding(step_data)


def get_embedding_dim() -> int:
    """Return the dimensionality of the state embedding vector."""
    return (
        len(APP_DOMAINS)          # app domain one-hot
        + NUM_CONTROL_TYPE_BINS   # control type distribution
        + len(CANONICAL_TABS) + 1 # tab presence
        + 1                       # dialog indicator
        + 1                       # total controls (log)
    )


# ============================================================================
# Trajectory processing
# ============================================================================

@dataclass
class TransitionRecord:
    """A single state transition from a trajectory."""
    state_id: StateID
    next_state_id: StateID
    action_type: str
    action_function: str
    execution_id: str
    step_id: int
    # Per-step embeddings (actual, not hash-looked-up). Used for f_net training.
    src_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    dst_embedding: Optional[np.ndarray] = field(default=None, repr=False)


class GUI360TrajectoryProcessor:
    """Process GUI-360 trajectories into state transitions for graph building.

    Usage:
        processor = GUI360TrajectoryProcessor()
        transitions = processor.process_trajectory_file("path/to/traj.jsonl")
        # Or process an entire directory:
        all_transitions = processor.process_directory("datasets/GUI-360/train/data")
    """

    def __init__(self, granularity: str = "fine"):
        """
        Args:
            granularity: "fine" (6,511 states) or "coarse" (app+tabs+dialog only,
                         ~600 states, better connectivity for graph analysis).
        """
        self.granularity = granularity
        self._state_registry: dict[str, StateID] = {}  # hash -> StateID
        self._embedding_cache: dict[str, np.ndarray] = {}  # hash -> embedding
        self._transition_count = 0

    @property
    def num_unique_states(self) -> int:
        return len(self._state_registry)

    @property
    def num_transitions(self) -> int:
        return self._transition_count

    def get_all_state_ids(self) -> list[StateID]:
        return list(self._state_registry.values())

    def get_state_embeddings(self) -> dict[str, np.ndarray]:
        """Return cached state embeddings keyed by hash."""
        return dict(self._embedding_cache)

    def _register_state(
        self, state_id: StateID, embedding: Optional[np.ndarray] = None
    ) -> StateID:
        """Register a state and optionally cache its embedding."""
        if state_id.hash_value not in self._state_registry:
            self._state_registry[state_id.hash_value] = state_id
        if embedding is not None and state_id.hash_value not in self._embedding_cache:
            self._embedding_cache[state_id.hash_value] = embedding
        return self._state_registry[state_id.hash_value]

    def process_step(
        self, raw_step: dict[str, Any], compute_embedding: bool = True
    ) -> tuple[GUI360StepData, StateID, Optional[np.ndarray]]:
        """Process a single raw step into structured data, state ID, and embedding."""
        step_data = GUI360StepData.from_raw(raw_step)
        state_id = extract_state_id(step_data, granularity=self.granularity)
        embedding = extract_state_embedding(step_data) if compute_embedding else None
        self._register_state(state_id, embedding)
        return step_data, state_id, embedding

    def process_trajectory_file(
        self,
        filepath: str | Path,
        compute_embeddings: bool = True,
    ) -> list[TransitionRecord]:
        """Process a single JSONL trajectory file into transition records.

        Each JSONL file has one JSON object per line, each line representing
        one step in the trajectory.

        Args:
            filepath: Path to a .jsonl trajectory file.
            compute_embeddings: Whether to also compute dense embeddings.

        Returns:
            List of TransitionRecords (consecutive step pairs).
        """
        filepath = Path(filepath)
        transitions = []

        with open(filepath, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            return transitions

        prev_step_data, prev_state_id, prev_emb = self.process_step(
            json.loads(lines[0]), compute_embeddings
        )

        for line in lines[1:]:
            raw = json.loads(line)
            curr_step_data, curr_state_id, curr_emb = self.process_step(
                raw, compute_embeddings
            )

            transitions.append(TransitionRecord(
                state_id=prev_state_id,
                next_state_id=curr_state_id,
                action_type=prev_step_data.action_type,
                action_function=prev_step_data.action_function,
                execution_id=prev_step_data.execution_id,
                step_id=prev_step_data.step_id,
                src_embedding=prev_emb,
                dst_embedding=curr_emb,
            ))
            self._transition_count += 1

            prev_step_data = curr_step_data
            prev_state_id = curr_state_id
            prev_emb = curr_emb

        return transitions

    def process_directory(
        self,
        data_dir: str | Path,
        compute_embeddings: bool = True,
        max_files: Optional[int] = None,
    ) -> list[TransitionRecord]:
        """Process all trajectory files under a directory tree.

        Recursively finds all .jsonl files under data_dir.

        Args:
            data_dir: Root directory (e.g., "datasets/GUI-360/train/data").
            compute_embeddings: Whether to compute dense embeddings.
            max_files: Optional limit on number of files to process.

        Returns:
            All transition records.
        """
        data_dir = Path(data_dir)
        files = sorted(data_dir.rglob("*.jsonl"))

        if max_files is not None:
            files = files[:max_files]

        logger.info(f"Processing {len(files)} trajectory files from {data_dir}")

        all_transitions = []
        for i, fpath in enumerate(files):
            try:
                transitions = self.process_trajectory_file(
                    fpath, compute_embeddings
                )
                all_transitions.extend(transitions)
            except Exception as e:
                logger.warning(f"Failed to process {fpath}: {e}")
                continue

            if (i + 1) % 1000 == 0:
                logger.info(
                    f"Processed {i+1}/{len(files)} files, "
                    f"{self.num_unique_states} unique states, "
                    f"{self.num_transitions} transitions"
                )

        logger.info(
            f"Done: {len(files)} files, "
            f"{self.num_unique_states} unique states, "
            f"{self.num_transitions} transitions"
        )
        return all_transitions

    def build_adjacency_dict(
        self, transitions: list[TransitionRecord]
    ) -> dict[str, dict[str, int]]:
        """Build an adjacency dict (transition count matrix) from transitions.

        Returns:
            Dict mapping state_hash -> {next_state_hash -> count}.
        """
        adj: dict[str, dict[str, int]] = {}
        for t in transitions:
            src = t.state_id.hash_value
            dst = t.next_state_id.hash_value
            if src not in adj:
                adj[src] = {}
            adj[src][dst] = adj[src].get(dst, 0) + 1
        return adj

    def get_statistics(self, transitions: list[TransitionRecord]) -> dict:
        """Compute graph statistics."""
        adj = self.build_adjacency_dict(transitions)
        num_edges = sum(len(neighbors) for neighbors in adj.values())

        # In/out degree distribution
        out_degrees = [len(neighbors) for neighbors in adj.values()]
        in_degree_counter: Counter = Counter()
        for neighbors in adj.values():
            for dst in neighbors:
                in_degree_counter[dst] += 1

        n = self.num_unique_states
        return {
            "num_states": n,
            "num_transitions": len(transitions),
            "num_unique_edges": num_edges,
            "graph_density": num_edges / (n * (n - 1)) if n > 1 else 0,
            "avg_out_degree": np.mean(out_degrees) if out_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "avg_in_degree": np.mean(list(in_degree_counter.values())) if in_degree_counter else 0,
            "max_in_degree": max(in_degree_counter.values()) if in_degree_counter else 0,
        }

    def save(self, output_dir: str | Path) -> None:
        """Save processor state (state registry, embeddings, stats) to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save state registry
        registry = {
            h: sid.to_dict() for h, sid in self._state_registry.items()
        }
        with open(output_dir / "state_registry.json", "w") as f:
            json.dump(registry, f, indent=2)

        # Save embeddings as numpy
        if self._embedding_cache:
            hashes = sorted(self._embedding_cache.keys())
            embeddings = np.stack([self._embedding_cache[h] for h in hashes])
            np.savez(
                output_dir / "state_embeddings.npz",
                hashes=np.array(hashes),
                embeddings=embeddings,
            )

        logger.info(f"Saved {len(registry)} states to {output_dir}")

    def load(self, output_dir: str | Path) -> None:
        """Load previously saved processor state."""
        output_dir = Path(output_dir)

        # Load state registry
        with open(output_dir / "state_registry.json", "r") as f:
            registry = json.load(f)

        for h, data in registry.items():
            sid = StateID(
                app_domain=data["app_domain"],
                active_tab_signature=data["active_tab_signature"],
                dialog_state=data["dialog_state"],
                control_fingerprint=data["control_fingerprint"],
            )
            self._state_registry[h] = sid

        # Load embeddings
        emb_path = output_dir / "state_embeddings.npz"
        if emb_path.exists():
            npz = np.load(emb_path, allow_pickle=True)
            hashes = npz["hashes"]
            embeddings = npz["embeddings"]
            for h, emb in zip(hashes, embeddings):
                self._embedding_cache[str(h)] = emb

        logger.info(
            f"Loaded {len(self._state_registry)} states from {output_dir}"
        )


def save_transitions(
    transitions: list[TransitionRecord],
    output_path: str | Path,
) -> None:
    """Save transition records to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for t in transitions:
            record = {
                "state_hash": t.state_id.hash_value,
                "next_state_hash": t.next_state_id.hash_value,
                "action_type": t.action_type,
                "action_function": t.action_function,
                "execution_id": t.execution_id,
                "step_id": t.step_id,
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved {len(transitions)} transitions to {output_path}")


def load_transitions(input_path: str | Path) -> list[dict]:
    """Load transition records from a JSONL file (as dicts)."""
    input_path = Path(input_path)
    records = []
    with open(input_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records
