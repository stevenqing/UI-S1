# GUI-360 A11y (Accessibility) Support for SFT Training

This document describes how to add accessibility (a11y) support to GUI-360 SFT training, which significantly improves grounding and action prediction accuracy.

## Overview

### What is A11y Support?

A11y support leverages accessibility metadata from the operating system (UI Automation API on Windows) to provide:

1. **Set-of-Mark (SoM) Annotated Screenshots** - Screenshots with numbered labels on interactive elements
2. **Element List in Prompt** - Semantic information about each labeled element
3. **Element ID-based Actions** - Using element IDs instead of pixel coordinates

### Performance Improvement

According to the GUI-360 paper, a11y support provides significant accuracy gains:

| Metric | Visual-only | Visual+A11y |
|--------|-------------|-------------|
| GUI Grounding | 35.78% | **82.30%** |
| Action Prediction | 17.52% | **50.08%** |

## Data Format Comparison

### Non-A11y Format

**Input:**
```
<image>
Instruction: Click on the Insert menu...
Actions: click(coordinate=[100, 200], button='left')
```

**Output:**
```json
{
  "function": "click",
  "args": {"coordinate": [147, 71]},
  "status": "CONTINUE"
}
```

### A11y Format

**Input:**
```
<image>
Instruction: Click on the Insert menu...

The elements on the screen are labeled with numbers:
[1] Button "AutoSave" (position: 98,23)
[2] Button "Save" (position: 169,23)
[13] MenuItem "Insert" (position: 147,71)
...

Actions: click(element_id=13, button='left')
```

**Output:**
```json
{
  "function": "click",
  "args": {"element_id": 13},
  "status": "CONTINUE"
}
```

## Implementation

### 1. Data Preparation Script

Location: `scripts/GUI_360/prepare_gui360_sft_parquet_a11y.py`

This script processes raw GUI-360 trajectories and generates a11y-formatted training data:

```bash
# Generate training data
python scripts/GUI_360/prepare_gui360_sft_parquet_a11y.py \
    --input-dir datasets/GUI-360/train/data \
    --output train_GUI_360/data/gui360_train_sft_a11y.parquet \
    --image-base-dir datasets/GUI-360/train/images \
    --apps word excel ppt

# Generate evaluation data
python scripts/GUI_360/prepare_gui360_sft_parquet_a11y.py \
    --input-dir datasets/GUI-360/test/data \
    --output train_GUI_360/data/gui360_eval_sft_a11y.parquet \
    --image-base-dir datasets/GUI-360/test/images \
    --apps word excel ppt
```

### 2. Key Components

#### Screenshot Selection

The script uses annotated screenshots with SoM labels:

```python
# Use annotated screenshot instead of clean one
if use_annotated_screenshot:
    screenshot_path = step.get('screenshot_annotated')
else:
    screenshot_path = step.get('screenshot_clean')
```

#### Element List Formatting

Each interactive element is formatted with its label, type, text, and position:

```python
def format_element_list(controls: List[Dict]) -> str:
    """Format control list for prompt."""
    lines = []
    for ctrl in controls:
        label = ctrl.get('label', '?')
        text = ctrl.get('control_text', '')
        ctrl_type = ctrl.get('control_type', 'Unknown')
        rect = ctrl.get('control_rect', [])

        center_x = (rect[0] + rect[2]) // 2
        center_y = (rect[1] + rect[3]) // 2
        lines.append(f"[{label}] {ctrl_type} \"{text}\" (position: {center_x},{center_y})")

    return "\n".join(lines)
```

#### Coordinate to Element ID Mapping

The script maps pixel coordinates to element IDs:

```python
def find_element_id_for_coordinate(controls, coordinate, tolerance=20):
    """Find element ID that contains or is closest to the coordinate."""
    target_x, target_y = coordinate[0], coordinate[1]

    for ctrl in controls:
        label = ctrl.get('label')
        rect = ctrl.get('control_rect', [])
        left, top, right, bottom = rect

        # Check if coordinate is inside the element (with tolerance)
        if left - tolerance <= target_x <= right + tolerance:
            if top - tolerance <= target_y <= bottom + tolerance:
                return label

    return None
```

### 3. Training Configuration

Location: `train_GUI_360/config/gui360_sft_a11y.yaml`

Key configuration differences from non-a11y training:

```yaml
# Data configuration
data:
  train_files: train_GUI_360/data/gui360_train_sft_a11y.parquet
  val_files: train_GUI_360/data/gui360_eval_sft_a11y.parquet

# Optimizer - Paper recommends {1e-5, 5e-6, 1e-6}
optim:
  lr: 1e-5

# Fewer epochs needed with a11y
trainer:
  total_epochs: 3
  project_name: gui360-sft-a11y
```

### 4. Submit Training Job

```bash
sbatch train_GUI_360/sft_scripts/train_gui360_sft_a11y.slurm
```

## Raw Data Structure

GUI-360 raw trajectories contain a11y information in `step.control_infos`:

```json
{
  "step": {
    "screenshot_clean": "path/to/clean.png",
    "screenshot_annotated": "path/to/annotated.png",
    "control_infos": {
      "merged_controls_info": [
        {
          "label": 1,
          "control_type": "Button",
          "control_text": "Save",
          "control_rect": [100, 50, 150, 80]
        },
        ...
      ]
    },
    "action": {
      "function": "click",
      "coordinate_x": 125,
      "coordinate_y": 65
    }
  }
}
```

## Prompt Template

The a11y system prompt includes element information and element_id-based actions:

```
You are a helpful assistant. Given a screenshot of the current screen with
labeled elements, user instruction and history of actions, you need to decide
the next action to take.

The instruction is:
{instruction}

The elements on the screen are labeled with numbers:
{element_list}

The actions supported are:
- click(element_id: int, button: str = 'left', double: bool = False)
- type(element_id: int, text: str, clear_current_text: bool = False)
- drag(start_element_id: int, end_element_id: int, ...)
...

Output your action in JSON format:
{
  "function": "click",
  "args": {"element_id": 13},
  "status": "CONTINUE"
}
```

## Benefits of A11y Approach

1. **Reduced Visual Grounding Burden** - Model doesn't need to predict exact pixel coordinates
2. **Semantic Understanding** - Element names provide context about UI components
3. **Better Generalization** - Element IDs are more transferable across screen resolutions
4. **Error Reduction** - No coordinate prediction errors (out-of-bounds clicks)

## Comparison: A11y vs Non-A11y Training

| Aspect | Non-A11y | A11y |
|--------|----------|------|
| Screenshot | Clean | SoM Annotated |
| Input | Instruction only | Instruction + Element list |
| Output | `coordinate=[x, y]` | `element_id=N` |
| Accuracy | ~50% | ~82% |
| Training epochs | 5 | 3 |
| Learning rate | 2e-6 | 1e-5 |

## Files Reference

| File | Description |
|------|-------------|
| `scripts/GUI_360/prepare_gui360_sft_parquet_a11y.py` | Data preparation script |
| `train_GUI_360/config/gui360_sft_a11y.yaml` | Training configuration |
| `train_GUI_360/sft_scripts/train_gui360_sft_a11y.slurm` | SLURM training script |
| `train_GUI_360/data/gui360_train_sft_a11y.parquet` | Training data (generated) |
| `train_GUI_360/data/gui360_eval_sft_a11y.parquet` | Eval data (generated) |

## References

- GUI-360 Paper: [arXiv:2511.04307](https://arxiv.org/abs/2511.04307)
- Set-of-Mark (SoM): [Yang et al., 2023](https://arxiv.org/abs/2310.11441)
- GUI-360 Dataset: [Hugging Face](https://huggingface.co/datasets/vyokky/GUI-360)
