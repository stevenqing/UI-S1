# V20 Phase 1: LoRA Specialty Training from V15 SFT

## Overview
Train 5 LoRA specialty adapters on top of V15 SFT checkpoint-272 (TSR=21.9%, StepSR=69%).
Use base V15 model (no LoRA) as 6th generalist agent. Compute oracle TSR across all 6.

## Step 1: Prepare Specialty Data Subsets
**Script:** `v20/phase1/prepare_specialty_data.py`
**Input:** V15 balanced training data (`v15_gui_360/data/gui360_balanced_train.jsonl`, 17264 examples)
**Output:** 5 JSONL files in `v20/phase1/data/`

Data subsets (biased sampling from V15's 17K examples):
1. **type_specialist** (~4500): ALL 3269 type + random 800 click + 500 drag
2. **drag_specialist** (~3500): ALL 2352 drag + random 800 click + 400 type
3. **spreadsheet_specialist** (~4700): ALL 4724 spreadsheet-task examples
4. **powerpoint_specialist** (~4000): Random 4000 from 6989 PPT examples
5. **word_specialist** (~4000): Random 4000 from 5543 Word examples

Format: V15's EXACT ShareGPT format (no prompt modification). Diversity comes from data bias only.

## Step 2: Register Datasets
Add 5+5 entries (train+val) to `train_GUI_360/llamafactory/data/dataset_info.json`.
Val sets: 10% held out from each specialty subset.

## Step 3: Create LoRA Training Configs
**5 YAML configs** in `v20/phase1/configs/`

Key parameters:
- `model_name_or_path`: `train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272`
- `finetuning_type`: lora
- `lora_rank`: 32
- `lora_alpha`: 64
- `lora_target`: q_proj,k_proj,v_proj,o_proj
- `freeze_vision_tower`: true
- `learning_rate`: 1e-5
- `num_train_epochs`: 2
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 4 (effective BS=16 on 4 GPUs)
- `deepspeed`: ZeRO-2 config (more efficient for LoRA than ZeRO-3)
- `cutoff_len`: 8192
- `template`: qwen2_vl
- `save_steps`: 50, `save_total_limit`: 3
- Output: `v20/phase1/output/lora_{agent}/`

## Step 4: Create DeepSpeed ZeRO-2 Config
`v20/phase1/configs/ds_z2_config.json` - standard ZeRO-2 for LoRA training.

## Step 5: Create Slurm Training Script
**Single parameterized script**: `v20/phase1/scripts/train_lora_specialist.slurm`
- 1 node × 4 GPUs (avoid NCCL cross-node issues)
- `--export=AGENT=type_specialist`
- Submit 5 jobs in parallel

## Step 6: Evaluation
**Script:** `v20/phase1/eval_lora_specialists.py`

Option A (efficient - single vLLM instance):
- Start vLLM with `--enable-lora --lora-modules type_specialist=path1 drag_specialist=path2 ...`
- Evaluate all 6 agents (5 LoRA + base) through same server
- Use `model=<agent_name>` in API calls to route to correct adapter

Option B (simpler - sequential):
- Merge each LoRA adapter with base model, evaluate sequentially
- Same approach as phase0 but with merged models

Recommended: Option A (much faster, single GPU allocation)

**Eval format:** V15's standard prompt, GT-history mode, 1000 balanced test episodes.
**Compute:** Oracle TSR across 5 LoRA specialists + base V15 model.

## Expected Results
- Each LoRA specialist: StepSR ~65-72% (close to V15's 69%, biased toward specialty)
- Base V15 model: StepSR 69%, TSR 22%
- Key metric: pairwise correlation should be MUCH lower than phase0 (0.5-0.6)
- Target oracle TSR: ≥ 30% (vs phase0's 22.4%)
- Target oracle StepSR: ≥ 80%

## Disk Space
- Each LoRA adapter: ~50-100MB (vs 16GB for full model)
- Total new storage: ~500MB for 5 adapters + ~500MB data
- No need to duplicate the 16GB base model

## Files to Create
1. `v20/phase1/prepare_specialty_data.py` - data preparation
2. `v20/phase1/configs/qwen25vl_lora_{agent}.yaml` × 5 - training configs
3. `v20/phase1/configs/ds_z2_config.json` - DeepSpeed config
4. `v20/phase1/scripts/train_lora_specialist.slurm` - training script
5. `v20/phase1/eval_lora_specialists.py` - evaluation script
6. `v20/phase1/scripts/eval_lora_specialists.slurm` - eval Slurm script
7. Update `dataset_info.json` with new datasets
