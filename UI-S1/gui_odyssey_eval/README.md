# GUI-Odyssey Evaluation Pipeline

Cross-app mobile navigation benchmark: 8,334 episodes, 127k screenshots, 6 task categories.

## Pipeline Overview

```
annotations/*.json + splits/*.json
        │
        ▼  convert_to_eval_format.py
gui_odyssey_{split}_test.jsonl  (AC-compatible)
        │
        ▼  eval_ar_trajectory.py  (vLLM server)
trajectory_results.jsonl + summary.json
```

## Files

| File | Description |
|------|-------------|
| `convert_to_eval_format.py` | GUI-Odyssey → AC-compatible JSONL converter |
| `odyssey_action_matching.py` | Action matching for [0,1000] coordinate system |
| `eval_ar_trajectory.py` | AR trajectory evaluation (main script) |
| `eval_odyssey.slurm` | SLURM job script (vLLM + eval) |
| `GUIOdyssey_action_matching.py` | Original action matching (reused for text/Levenshtein) |

## Quick Start

### 1. Convert dataset

```bash
python gui_odyssey_eval/convert_to_eval_format.py \
    --data_dir datasets/GUI-Odyssey \
    --split random_split \
    --subset test
```

### 2. Run evaluation (SLURM)

```bash
# Default: Qwen2.5-VL-7B on random_split
sbatch gui_odyssey_eval/eval_odyssey.slurm

# Custom model/split
sbatch --export=MODEL_PATH=/path/to/model,SPLIT=app_split \
    gui_odyssey_eval/eval_odyssey.slurm
```

### 3. Run evaluation (manual)

```bash
# Start vLLM server first, then:
python gui_odyssey_eval/eval_ar_trajectory.py \
    --jsonl_file datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl \
    --model_name Qwen2.5-VL-7B-Instruct \
    --output_dir gui_odyssey_eval/results/base \
    --max_workers 4
```

## Evaluation Metrics

### Step-Level: Action Matching

| Action | Matching Criteria |
|--------|------------------|
| click / long_press | sam2_bbox containment (1.2x enlarged) OR distance ≤ 140 in [0,1000] space |
| swipe (scroll) | Direction match (UP/DOWN/LEFT/RIGHT) |
| type | ANLS ≥ 0.5 (Levenshtein distance) |
| system_button | Button name match (Home/Back/Menu) |
| terminate | Status match (success/failure) |

### Trajectory-Level

| Metric | Description |
|--------|-------------|
| **TSR** | Task Success Rate — all steps correct |
| **Avg Progress** | Mean(correct_steps / total_steps) per episode |
| **Scattered Progress** | Total correct / total steps across all episodes |

### Breakdowns

- **Per-category**: Web_Shopping, Social_Sharing, Information_Management, Multi_Apps, General_Tool, Life_Assistant
- **Per-device**: Pixel 7 Pro, Pixel 8 Pro, Pixel Tablet, Small Phone, etc.
- **Per-length-bucket**: short(1-3), medium(4-7), long(8-15), vlong(16+)
- **Per-action-type**: click, long_press, swipe, type, system_button, terminate

## Coordinate System

- **Ground truth**: [0, 1000] normalized coordinates
- **Model output**: Resized pixel space → converted to [0,1000] via `pred / resized_dim * 1000`
- **sam2_bbox**: [x1, y1, x2, y2] in [0, 1000] space

## Dataset Splits

| Split | Train | Test | Description |
|-------|------:|-----:|-------------|
| random_split | 6,668 | 1,666 | Random 4:1 split |
| app_split | — | — | Disjoint apps |
| device_split | — | — | Pixel Tablet vs others |
| task_split | — | — | Disjoint meta-tasks |

## Architecture

Reuses the AndroidControl evaluation infrastructure:
- `x/data/agent/json.py` — `JsonFormat.gen_next_round()` for message construction
- `x/qwen/data_format.py` — `slim_messages()` for history image limiting
- `evaluation/qwenvl_utils.py` — `call_mobile_agent_vllm()`, `find_last_image_ele()`
