# Base-Derived Memory Agent Full-SFT Training Package

This package contains the runnable plan and training entry points for base-derived memory-specialized proposal agents.

If another coding agent is taking over on a different machine, start with `AGENT_HANDOFF.md`. It contains the required context, what not to do, validation commands, launch commands, and the two-stage evaluation rule.

## Research Goal

Current candidate sources are prompt variants of the same UI-S1 SFT checkpoint, so their errors remain correlated. This package trains separate full-parameter memory agents from the base Qwen2.5-VL model, each using a different proposal dataset. The goal is not standalone policy TSR. The goal is candidate-pool diversity:

1. Train a memory-specialized proposal agent from base Qwen2.5-VL.
2. Use it as a candidate source.
3. Diagnose coverage with GT-history candidate generation.
4. Deploy autoregressively with selector/aggregator.
5. Measure oracle TSR gain and final trajectory TSR.

## Agents

- `type_recovery`: type / field-focus / text-entry recovery.
- `click_recovery`: alternative grounded click targets.
- `swipe_navigation`: scroll, swipe, drag, navigation proposals.
- `minimal_next_step`: simple robust next-step proposals.
- `escape_finish`: modal, escape, finish guard proposals.
- `spreadsheet_formula`: spreadsheet, formula, cell/table editing proposals.

## Data

Generated source data lives in:

```text
v20/phase0/data/proposal_sft/per_agent_2000/
```

Each agent has:

```text
<agent>_train_sharegpt.jsonl  # 2000 examples
<agent>_dev_sharegpt.jsonl    # 300 examples
```

The examples are ShareGPT-style records with screenshot image paths:

```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "<tool_call>...</tool_call>"}
  ],
  "images": ["/absolute/path/to/step.png"]
}
```

## Configs

Full-parameter SFT configs are in `configs/`:

```text
qwen25vl_full_proposal_type_recovery.yaml
qwen25vl_full_proposal_click_recovery.yaml
qwen25vl_full_proposal_swipe_navigation.yaml
qwen25vl_full_proposal_minimal_next_step.yaml
qwen25vl_full_proposal_escape_finish.yaml
qwen25vl_full_proposal_spreadsheet_formula.yaml
```

They use:

```yaml
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: false
freeze_language_model: false
```

## Launch

Use the generic Slurm launcher:

```bash
sbatch --export=RUN_KEY=type_recovery scripts/train_full_memory_agent.slurm
sbatch --export=RUN_KEY=click_recovery scripts/train_full_memory_agent.slurm
sbatch --export=RUN_KEY=swipe_navigation scripts/train_full_memory_agent.slurm
sbatch --export=RUN_KEY=minimal_next_step scripts/train_full_memory_agent.slurm
sbatch --export=RUN_KEY=escape_finish scripts/train_full_memory_agent.slurm
sbatch --export=RUN_KEY=spreadsheet_formula scripts/train_full_memory_agent.slurm
```

The launcher expects the repository root at `/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1` by default. On another machine, edit `PROJECT_DIR`, conda path, CUDA path, and partition settings in `scripts/train_full_memory_agent.slurm`.

## Required Environment

- LLaMA-Factory CLI available as `llamafactory-cli`.
- Qwen2.5-VL base model available at the path configured by `model_name_or_path`.
- DeepSpeed ZeRO-3 config available at `train_GUI_360/llamafactory/ds_z3_config.json`.
- Screenshot paths in the JSONL files must exist on the target machine.

## Evaluation After Training

Do not treat GT-history as deployment. Use two-stage evaluation:

1. Offline coverage diagnostic: run each trained model as a candidate source with GT-history full generation and recompute oracle TSR.
2. Autoregressive deployment: each memory agent conditions on actual selected history, selector/aggregator chooses one action, and trajectory TSR is measured.

Current prompt-specialized UI-S1 pool oracle is 32.7%. A trained memory-agent pool must beat 32.7% offline and then improve autoregressive TSR to count as a successful research result.
