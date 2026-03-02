# DCP (Distributed Checkpoint) Fast Checkpoint Saving

## Overview

DCP (Distributed Checkpoint) enables fast parallel checkpoint saving for FSDP2 training. Instead of gathering the full model state to rank 0 (which takes ~54 minutes for a 7B model on 16 GPUs), DCP saves sharded checkpoints in parallel across all ranks.

**Speedup: 54 min → 3-5 min per checkpoint save (~10-15x faster)**

## Why DCP is Faster

| Method | How it works | Time (7B model, 16 GPUs) |
|--------|-------------|--------------------------|
| Full State Dict | All-gather to rank 0, then write | ~54 min |
| DCP | Each rank writes its shard in parallel | ~3-5 min |

## Usage

### Training (Automatic)

DCP is now the **default** for checkpoint saving. Just run training as before:

```bash
sbatch train_GUI_360/sft_scripts/train_gui360_sft_v4_a11y_thinking.slurm
```

Checkpoints will be saved in DCP format with a `.dcp_checkpoint` marker file.

### Resume from DCP Checkpoint

Set `resume_path` in your config:

```yaml
trainer:
  resume_path: /path/to/checkpoints/global_step_50
```

The trainer auto-detects DCP vs HuggingFace format.

### Convert DCP to HuggingFace Format

DCP checkpoints are sharded and require distributed loading. To use checkpoints for inference, convert to HuggingFace format:

```bash
# Multi-GPU conversion (recommended for 7B+ models)
torchrun --nproc_per_node=4 scripts/convert_dcp_checkpoint.py \
    --dcp-path train_GUI_360/checkpoints/gui360_sft_v4_a11y_thinking/global_step_50 \
    --hf-path train_GUI_360/checkpoints/gui360_sft_v4_a11y_thinking/global_step_50_hf \
    --model-path checkpoints/Qwen2.5-VL-7B-Instruct

# SLURM conversion job
srun --nodes=1 --ntasks-per-node=4 --gres=gpu:4 \
    python scripts/convert_dcp_checkpoint.py \
    --dcp-path ... --hf-path ... --model-path ...
```

## API Reference

### `save_checkpoint(step, use_dcp=True)`

Main checkpoint saving method.

```python
# Fast DCP save (default)
trainer.save_checkpoint(step=100)

# Slow HuggingFace format (for final checkpoint)
trainer.save_checkpoint(step=100, use_dcp=False)
```

### `load_checkpoint(path)`

Load checkpoint with auto-detection of format.

```python
step = trainer.load_checkpoint("/path/to/checkpoint")
# Returns: the training step from the checkpoint
```

### `convert_dcp_to_hf(dcp_path, hf_path)`

Convert DCP checkpoint to HuggingFace format (requires distributed environment).

```python
trainer.convert_dcp_to_hf(
    dcp_path="/path/to/dcp_checkpoint",
    hf_path="/path/to/hf_output"
)
```

## Checkpoint Directory Structure

### DCP Checkpoint
```
global_step_50/
├── .dcp_checkpoint          # Marker file indicating DCP format
├── .metadata                 # DCP metadata
├── __0_0.distcp             # Shard 0
├── __1_0.distcp             # Shard 1
├── ...                       # More shards
├── config.json              # Model config
├── tokenizer.json           # Tokenizer
└── tokenizer_config.json
```

### HuggingFace Checkpoint (after conversion)
```
global_step_50_hf/
├── config.json
├── model-00001-of-00007.safetensors
├── model-00002-of-00007.safetensors
├── ...
├── model.safetensors.index.json
├── tokenizer.json
└── tokenizer_config.json
```

## Implementation Details

### Files Modified

1. **`verl/trainer/fsdp_sft_trainer.py`**
   - Added `_save_checkpoint_dcp()` - parallel sharded saving
   - Added `_save_checkpoint_full()` - original HF format saving
   - Added `load_checkpoint()` - auto-detect and load
   - Added `_load_checkpoint_dcp()` - load sharded checkpoint
   - Added `_load_checkpoint_hf()` - load HF checkpoint
   - Added `convert_dcp_to_hf()` - conversion utility
   - Updated `fit()` - resume support with proper step skipping

2. **`scripts/convert_dcp_checkpoint.py`** (new)
   - Standalone script for DCP to HF conversion
   - Supports multi-GPU and multi-node conversion

### What DCP Saves

```python
state_dict = {
    "model": model_state_dict,      # Sharded model weights
    "optimizer": optim_state_dict,  # Sharded optimizer state
    "step": step,                   # Training step
    "lr_scheduler": lr_scheduler.state_dict(),
}
```

## Troubleshooting

### "Cannot load DCP checkpoint on single GPU"

DCP checkpoints require distributed loading. Use `torchrun` or convert to HF format first:

```bash
torchrun --nproc_per_node=4 scripts/convert_dcp_checkpoint.py ...
```

### "Checkpoint not found" when resuming

Make sure the checkpoint directory exists and contains either:
- `.dcp_checkpoint` marker (DCP format), or
- `model.safetensors` / `pytorch_model.bin` (HF format)

### "Shape mismatch" when loading

Ensure you're using the same number of GPUs and FSDP configuration as when the checkpoint was saved.

## Performance Comparison

Tested on 4 nodes × 4 GPUs (16 total) with Qwen2.5-VL-7B:

| Operation | Full State Dict | DCP |
|-----------|----------------|-----|
| Save checkpoint | 54 min | 3-5 min |
| Load checkpoint | 2-3 min | 1-2 min |
| Convert to HF | N/A | 54 min (one-time) |

**Recommendation:** Use DCP for all intermediate checkpoints during training. Convert only the final/best checkpoint to HF format for inference.
