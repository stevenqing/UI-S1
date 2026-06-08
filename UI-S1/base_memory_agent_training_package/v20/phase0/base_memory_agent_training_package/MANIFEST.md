# Package Manifest

## Top-Level Docs

- `README.md`: quick start and migration notes.
- `PLAN.md`: research plan, hypothesis, data generation, training, and evaluation.
- `AGENT_HANDOFF.md`: instructions for the coding agent on the target machine.
- `MANIFEST.md`: this file.

## Data Metadata

- `data/dataset_info.json`: LLaMA-Factory dataset registration for six per-agent datasets.
- `data/per_agent_2000_summary.json`: expected record counts and action-type distributions.

Actual JSONL data is generated or copied at:

```text
v20/phase0/data/proposal_sft/per_agent_2000/
```

## Configs

Full-parameter SFT configs:

- `configs/qwen25vl_full_proposal_type_recovery.yaml`
- `configs/qwen25vl_full_proposal_click_recovery.yaml`
- `configs/qwen25vl_full_proposal_swipe_navigation.yaml`
- `configs/qwen25vl_full_proposal_minimal_next_step.yaml`
- `configs/qwen25vl_full_proposal_escape_finish.yaml`
- `configs/qwen25vl_full_proposal_spreadsheet_formula.yaml`

## Scripts

- `scripts/generate_full_sft_configs.py`: regenerate the six full-SFT configs.
- `scripts/prepare_per_agent_data.sh`: regenerate per-agent 2000/300 ShareGPT datasets from full GT-history candidate outputs.
- `scripts/run_full_memory_agent_sft.sh`: direct runner for one agent with `RUN_KEY=<agent>`.
- `scripts/train_full_memory_agent.slurm`: Slurm launcher for one full-parameter memory agent.

## Supported RUN_KEY Values

- `type_recovery`
- `click_recovery`
- `swipe_navigation`
- `minimal_next_step`
- `escape_finish`
- `spreadsheet_formula`

## Typical Launch

```bash
sbatch --export=RUN_KEY=type_recovery v20/phase0/base_memory_agent_training_package/scripts/train_full_memory_agent.slurm
```

## Path Assumptions

Default `PROJECT_DIR` / `ROOT_DIR`:

```text
/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1
```

On another machine, edit:

- `PROJECT_DIR` in `scripts/train_full_memory_agent.slurm`
- `model_name_or_path` in `configs/*.yaml`
- conda path and CUDA path in the Slurm script
- dataset image paths if the screenshot cache moved
