"""Tests for state representation module (Task 1: GUI-360 only)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from verl.models.moe.state_representation import (
    GUI360StepData,
    GUI360TrajectoryProcessor,
    StateID,
    TransitionRecord,
    extract_state_embedding,
    extract_state_id,
    extract_state_id_from_raw,
    extract_state_embedding_from_raw,
    get_embedding_dim,
    save_transitions,
    load_transitions,
    _bucket_count,
    _build_control_fingerprint,
    _normalize_tab_signature,
)

# ============================================================================
# Fixtures: synthetic GUI-360 step data
# ============================================================================

def _make_raw_step(
    execution_id="excel_4_1000",
    app_domain="excel",
    step_id=1,
    total_steps=7,
    request="Create a table",
    window_title="test - Excel",
    merged_controls=None,
    dialog_names=None,
    action_type="GUI",
    action_function="click",
    status="CONTINUE",
):
    """Build a minimal raw JSONL step dict matching GUI-360 format."""
    if merged_controls is None:
        merged_controls = [
            {"control_type": "DataItem", "control_rect": [0, 0, 100, 20], "control_text": "A1", "source": "uia", "label": 1},
            {"control_type": "Button", "control_rect": [0, 0, 50, 30], "control_text": "Bold", "source": "uia", "label": 2},
            {"control_type": "TabItem", "control_rect": [58, 48, 113, 78], "control_text": "Home", "source": "uia", "label": 3},
            {"control_type": "TabItem", "control_rect": [114, 48, 167, 78], "control_text": "Insert", "source": "uia", "label": 4},
            {"control_type": "MenuItem", "control_rect": [0, 0, 80, 20], "control_text": "File", "source": "uia", "label": 5},
        ]

    ui_children = []
    if dialog_names:
        for name in dialog_names:
            ui_children.append({
                "id": "node_d",
                "name": name,
                "control_type": "Window",
                "rectangle": {"left": 100, "top": 100, "right": 400, "bottom": 400},
                "adjusted_rectangle": {"left": 100, "top": 100, "right": 400, "bottom": 400},
                "relative_rectangle": {"left": 0.1, "top": 0.1, "right": 0.4, "bottom": 0.4},
                "level": 1,
                "children": [],
            })

    return {
        "execution_id": execution_id,
        "app_domain": app_domain,
        "request": request,
        "template": "template1.xlsx",
        "step_id": step_id,
        "total_steps": total_steps,
        "evaluation": {"reason": "", "complete": "yes", "sub_scores": {}},
        "step": {
            "screenshot_clean": "images/clean.png",
            "screenshot_desktop": "images/desktop.png",
            "screenshot_annotated": "images/annotated.png",
            "screenshot_selected_controls": "images/selected.png",
            "ui_tree": {
                "id": "node_0",
                "name": window_title,
                "control_type": "Window",
                "rectangle": {"left": 0, "top": 0, "right": 1024, "bottom": 728},
                "adjusted_rectangle": {"left": 0, "top": 0, "right": 1024, "bottom": 728},
                "relative_rectangle": {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0},
                "level": 0,
                "children": ui_children,
            },
            "control_infos": {
                "application_windows_info": {
                    "control_type": "Window",
                    "control_rect": [0, 0, 1024, 728],
                    "control_text": window_title,
                    "source": "",
                },
                "uia_controls_info": merged_controls,
                "grounding_controls_info": [],
                "merged_controls_info": merged_controls,
            },
            "subtask": "Do something",
            "observation": "I see the screen",
            "thought": "I should click",
            "action": {
                "action_type": action_type,
                "control_test": "A1",
                "control_label": "1",
                "function": action_function,
                "args": {},
                "rectangle": {"left": 0, "top": 0, "right": 100, "bottom": 20},
                "coordinate_x": 50.0,
                "coordinate_y": 10.0,
                "desktop_rectangle": {"left": 0, "top": 0, "right": 100, "bottom": 20},
                "desktop_coordinate_x": 50.0,
                "desktop_coordinate_y": 10.0,
            },
            "status": status,
            "tags": ["action_prediction"],
        },
    }


# ============================================================================
# Tests: GUI360StepData parsing
# ============================================================================

class TestGUI360StepData:
    def test_from_raw_basic(self):
        raw = _make_raw_step()
        step = GUI360StepData.from_raw(raw)
        assert step.execution_id == "excel_4_1000"
        assert step.app_domain == "excel"
        assert step.step_id == 1
        assert step.total_steps == 7
        assert step.window_title == "test - Excel"
        assert step.action_type == "GUI"
        assert step.action_function == "click"
        assert step.status == "CONTINUE"

    def test_control_type_counts(self):
        raw = _make_raw_step()
        step = GUI360StepData.from_raw(raw)
        assert step.control_type_counts["DataItem"] == 1
        assert step.control_type_counts["Button"] == 1
        assert step.control_type_counts["TabItem"] == 2
        assert step.control_type_counts["MenuItem"] == 1
        assert step.total_controls == 5

    def test_tab_names_extracted(self):
        raw = _make_raw_step()
        step = GUI360StepData.from_raw(raw)
        assert "Home" in step.tab_names
        assert "Insert" in step.tab_names

    def test_dialog_names_empty(self):
        raw = _make_raw_step()
        step = GUI360StepData.from_raw(raw)
        assert step.dialog_names == []

    def test_dialog_names_present(self):
        raw = _make_raw_step(dialog_names=["Format Cells"])
        step = GUI360StepData.from_raw(raw)
        assert step.dialog_names == ["Format Cells"]

    def test_word_domain(self):
        raw = _make_raw_step(app_domain="word", window_title="doc - Word")
        step = GUI360StepData.from_raw(raw)
        assert step.app_domain == "word"


# ============================================================================
# Tests: State ID extraction
# ============================================================================

class TestExtractStateID:
    def test_deterministic(self):
        raw = _make_raw_step()
        sid1 = extract_state_id_from_raw(raw)
        sid2 = extract_state_id_from_raw(raw)
        assert sid1.hash_value == sid2.hash_value
        assert sid1 == sid2

    def test_different_apps_different_states(self):
        raw1 = _make_raw_step(app_domain="excel")
        raw2 = _make_raw_step(app_domain="word")
        sid1 = extract_state_id_from_raw(raw1)
        sid2 = extract_state_id_from_raw(raw2)
        assert sid1 != sid2

    def test_dialog_changes_state(self):
        raw1 = _make_raw_step()
        raw2 = _make_raw_step(dialog_names=["Format Cells"])
        sid1 = extract_state_id_from_raw(raw1)
        sid2 = extract_state_id_from_raw(raw2)
        assert sid1 != sid2

    def test_different_tabs_different_state(self):
        controls1 = [
            {"control_type": "TabItem", "control_rect": [0, 0, 50, 20], "control_text": "Home", "source": "uia", "label": 1},
            {"control_type": "TabItem", "control_rect": [50, 0, 100, 20], "control_text": "Insert", "source": "uia", "label": 2},
        ]
        controls2 = [
            {"control_type": "TabItem", "control_rect": [0, 0, 50, 20], "control_text": "Home", "source": "uia", "label": 1},
            {"control_type": "TabItem", "control_rect": [50, 0, 100, 20], "control_text": "Insert", "source": "uia", "label": 2},
            {"control_type": "TabItem", "control_rect": [100, 0, 150, 20], "control_text": "Data", "source": "uia", "label": 3},
        ]
        sid1 = extract_state_id_from_raw(_make_raw_step(merged_controls=controls1))
        sid2 = extract_state_id_from_raw(_make_raw_step(merged_controls=controls2))
        assert sid1 != sid2

    def test_hashable(self):
        raw = _make_raw_step()
        sid = extract_state_id_from_raw(raw)
        # Can be used in sets and dicts
        s = {sid}
        assert sid in s

    def test_to_dict(self):
        raw = _make_raw_step()
        sid = extract_state_id_from_raw(raw)
        d = sid.to_dict()
        assert "app_domain" in d
        assert "hash" in d
        assert d["app_domain"] == "excel"


# ============================================================================
# Tests: State embedding extraction
# ============================================================================

class TestExtractStateEmbedding:
    def test_correct_dimensionality(self):
        raw = _make_raw_step()
        emb = extract_state_embedding_from_raw(raw)
        assert emb.shape == (get_embedding_dim(),)

    def test_dtype_float32(self):
        raw = _make_raw_step()
        emb = extract_state_embedding_from_raw(raw)
        assert emb.dtype == np.float32

    def test_app_one_hot(self):
        emb_excel = extract_state_embedding_from_raw(_make_raw_step(app_domain="excel"))
        emb_word = extract_state_embedding_from_raw(_make_raw_step(app_domain="word"))
        # First 3 dims are app one-hot
        assert emb_excel[0] == 1.0  # excel = index 0
        assert emb_word[1] == 1.0   # word = index 1

    def test_different_states_different_embeddings(self):
        emb1 = extract_state_embedding_from_raw(_make_raw_step(app_domain="excel"))
        emb2 = extract_state_embedding_from_raw(_make_raw_step(app_domain="ppt"))
        assert not np.allclose(emb1, emb2)

    def test_dialog_changes_embedding(self):
        emb1 = extract_state_embedding_from_raw(_make_raw_step())
        emb2 = extract_state_embedding_from_raw(_make_raw_step(dialog_names=["Paragraph"]))
        assert not np.allclose(emb1, emb2)


# ============================================================================
# Tests: Helper functions
# ============================================================================

class TestHelpers:
    def test_bucket_count(self):
        assert _bucket_count(0) == 0
        assert _bucket_count(3) == 1
        assert _bucket_count(5) == 1
        assert _bucket_count(10) == 2
        assert _bucket_count(15) == 2
        assert _bucket_count(50) == 3
        assert _bucket_count(200) == 5
        assert _bucket_count(1000) == 6

    def test_normalize_tab_signature_canonical_order(self):
        tabs = ["Insert", "Home", "View"]
        sig = _normalize_tab_signature(tabs)
        assert sig == "Home,Insert,View"

    def test_normalize_tab_signature_filters_noncanonical(self):
        tabs = ["Home", "AI Perfect Assistant", "Insert"]
        sig = _normalize_tab_signature(tabs)
        assert "AI Perfect Assistant" not in sig
        assert sig == "Home,Insert"

    def test_build_control_fingerprint_deterministic(self):
        counts = {"DataItem": 359, "Button": 63, "MenuItem": 22}
        fp1 = _build_control_fingerprint(counts)
        fp2 = _build_control_fingerprint(counts)
        assert fp1 == fp2

    def test_build_control_fingerprint_bucketed(self):
        # Slightly different counts in same bucket should produce same fingerprint
        counts1 = {"DataItem": 359, "Button": 63}
        counts2 = {"DataItem": 360, "Button": 65}
        fp1 = _build_control_fingerprint(counts1)
        fp2 = _build_control_fingerprint(counts2)
        assert fp1 == fp2  # both are in same bucket


# ============================================================================
# Tests: Trajectory processing
# ============================================================================

class TestGUI360TrajectoryProcessor:
    def _make_trajectory_file(self, tmp_path, num_steps=5):
        """Create a temporary JSONL trajectory file."""
        filepath = tmp_path / "test_traj.jsonl"
        with open(filepath, "w") as f:
            for i in range(num_steps):
                raw = _make_raw_step(
                    step_id=i + 1,
                    total_steps=num_steps,
                    action_function="click" if i % 2 == 0 else "type",
                    dialog_names=["Format Cells"] if i == 2 else None,
                )
                f.write(json.dumps(raw) + "\n")
        return filepath

    def test_process_trajectory_file(self, tmp_path):
        filepath = self._make_trajectory_file(tmp_path, num_steps=5)
        processor = GUI360TrajectoryProcessor()
        transitions = processor.process_trajectory_file(filepath)
        assert len(transitions) == 4  # 5 steps -> 4 transitions
        assert processor.num_transitions == 4

    def test_process_detects_state_changes(self, tmp_path):
        filepath = self._make_trajectory_file(tmp_path, num_steps=5)
        processor = GUI360TrajectoryProcessor()
        transitions = processor.process_trajectory_file(filepath)
        # Step 2->3 introduces a dialog, so states should differ
        t_with_dialog = transitions[1]  # step 2 -> step 3
        assert t_with_dialog.state_id != t_with_dialog.next_state_id

    def test_unique_states(self, tmp_path):
        filepath = self._make_trajectory_file(tmp_path, num_steps=5)
        processor = GUI360TrajectoryProcessor()
        processor.process_trajectory_file(filepath)
        # Most steps are identical (same controls), step 3 has a dialog -> 2 unique states
        assert processor.num_unique_states == 2

    def test_embeddings_cached(self, tmp_path):
        filepath = self._make_trajectory_file(tmp_path, num_steps=3)
        processor = GUI360TrajectoryProcessor()
        processor.process_trajectory_file(filepath, compute_embeddings=True)
        embeddings = processor.get_state_embeddings()
        assert len(embeddings) > 0
        for h, emb in embeddings.items():
            assert emb.shape == (get_embedding_dim(),)

    def test_build_adjacency_dict(self, tmp_path):
        filepath = self._make_trajectory_file(tmp_path, num_steps=5)
        processor = GUI360TrajectoryProcessor()
        transitions = processor.process_trajectory_file(filepath)
        adj = processor.build_adjacency_dict(transitions)
        assert len(adj) > 0
        # Check that all edges have positive counts
        for src, neighbors in adj.items():
            for dst, cnt in neighbors.items():
                assert cnt > 0

    def test_get_statistics(self, tmp_path):
        filepath = self._make_trajectory_file(tmp_path, num_steps=5)
        processor = GUI360TrajectoryProcessor()
        transitions = processor.process_trajectory_file(filepath)
        stats = processor.get_statistics(transitions)
        assert stats["num_states"] == 2
        assert stats["num_transitions"] == 4
        assert stats["graph_density"] > 0

    def test_save_and_load(self, tmp_path):
        filepath = self._make_trajectory_file(tmp_path, num_steps=5)
        processor = GUI360TrajectoryProcessor()
        processor.process_trajectory_file(filepath)

        save_dir = tmp_path / "saved"
        processor.save(save_dir)

        # Load into fresh processor
        processor2 = GUI360TrajectoryProcessor()
        processor2.load(save_dir)
        assert processor2.num_unique_states == processor.num_unique_states

    def test_process_directory(self, tmp_path):
        # Create a directory tree mimicking GUI-360 structure
        data_dir = tmp_path / "data" / "excel" / "search" / "success"
        data_dir.mkdir(parents=True)
        for i in range(3):
            filepath = data_dir / f"traj_{i}.jsonl"
            with open(filepath, "w") as f:
                for j in range(4):
                    raw = _make_raw_step(
                        execution_id=f"excel_{i}",
                        step_id=j + 1,
                        total_steps=4,
                    )
                    f.write(json.dumps(raw) + "\n")

        processor = GUI360TrajectoryProcessor()
        transitions = processor.process_directory(tmp_path / "data")
        assert len(transitions) == 9  # 3 files * 3 transitions each


# ============================================================================
# Tests: Transition save/load
# ============================================================================

class TestTransitionIO:
    def test_save_and_load(self, tmp_path):
        sid1 = StateID("excel", "Home,Insert", "none", "DataItem:6")
        sid2 = StateID("excel", "Home,Insert", "Format Cells", "DataItem:6,Button:2")

        transitions = [
            TransitionRecord(sid1, sid2, "GUI", "click", "test_1", 1),
            TransitionRecord(sid2, sid1, "GUI", "click", "test_1", 2),
        ]

        filepath = tmp_path / "transitions.jsonl"
        save_transitions(transitions, filepath)
        loaded = load_transitions(filepath)

        assert len(loaded) == 2
        assert loaded[0]["state_hash"] == sid1.hash_value
        assert loaded[0]["next_state_hash"] == sid2.hash_value


# ============================================================================
# Tests: Integration with real GUI-360 data (skip if not available)
# ============================================================================

REAL_DATA_DIR = Path("/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/train/data")


@pytest.mark.skipif(
    not REAL_DATA_DIR.exists(),
    reason="GUI-360 dataset not available"
)
class TestRealGUI360Data:
    def test_process_single_real_file(self):
        """Process a real GUI-360 trajectory file."""
        files = list(REAL_DATA_DIR.rglob("*.jsonl"))
        assert len(files) > 0, "No JSONL files found"

        processor = GUI360TrajectoryProcessor()
        transitions = processor.process_trajectory_file(files[0])

        assert len(transitions) >= 0  # could be 0 if single-step
        assert processor.num_unique_states >= 1

    def test_process_sample_of_real_data(self):
        """Process a small sample of real trajectories."""
        processor = GUI360TrajectoryProcessor()
        transitions = processor.process_directory(
            REAL_DATA_DIR, max_files=50
        )

        stats = processor.get_statistics(transitions)
        print(f"\nReal data stats (50 files):")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        assert stats["num_states"] > 1
        assert stats["num_transitions"] > 0

    def test_real_embeddings_valid(self):
        """Verify embeddings from real data have expected properties."""
        files = list(REAL_DATA_DIR.rglob("*.jsonl"))[:10]
        processor = GUI360TrajectoryProcessor()

        for f in files:
            processor.process_trajectory_file(f)

        embeddings = processor.get_state_embeddings()
        assert len(embeddings) > 0

        for h, emb in embeddings.items():
            assert emb.shape == (get_embedding_dim(),)
            assert np.isfinite(emb).all()
            # App one-hot should sum to 1
            assert np.isclose(emb[:3].sum(), 1.0)
