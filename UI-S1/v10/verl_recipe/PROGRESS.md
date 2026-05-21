# v10 verl Recipe — Implementation Progress

## Status: Week 1 — FSDP2 Compatibility Validated

### Completed (2026-04-19)

#### 1. Core Module Implementation
- `coop_fsdp_workers.py` — CoopActorRolloutRefWorker with dual LoRA adapter
- `coop_ray_trainer.py` — CoopRayTrainer with dual-phase fit loop
- `coop_dataset.py` — JSONL→parquet conversion + prompt templates
- `reward_fn.py` — Reward function (grounder + actor scores)
- `main_coop_grpo.py` — Ray + Hydra entry point
- `config/coop_grpo_ac.yaml` — AndroidControl training config

#### 2. Data Preparation
- AC training data: 6,482 samples → `data/ac_train.parquet`
- AC validation data: 54 samples → `data/ac_val.parquet`
- Prompt format: `<image>` placeholder style (compatible with verl RLHFDataset)

#### 3. FSDP2 + PEFT Multi-Adapter Compatibility Test
**Result: ALL PASS** (job 3999514)

| Test | Result |
|------|--------|
| FSDP2 wrapping with dual LoRA | PASS |
| `set_adapter()` switching | PASS (logit diff=18.875) |
| Gradient isolation | PASS (only active adapter gets grads) |
| `disable_adapter_layers()` ref policy | PASS (base model output) |
| `requires_grad` restoration | PASS (with manual fix) |

**Key findings:**
1. LoRA params must be cast to bf16 before FSDP2 (uniform dtype requirement)
2. `enable_adapter_layers()` only re-enables active adapter → need manual `requires_grad=True`
3. `fsdp_config` needs `{"wrap_policy": {}}` not `None`

#### 4. Scripts
- `scripts/test_fsdp2_adapter.slurm` — FSDP2 compatibility test
- `scripts/train_coop_ac.slurm` — Full training launch
- `scripts/prepare_ac_data.sh` — Data preparation

### Next Steps

#### Week 1 (remaining)
- [ ] End-to-end dry run with small batch (1-2 samples, 1 step)
- [ ] Verify dual-phase rollout (grounder → decode → actor prompt → actor)
- [ ] Verify reward computation with real data
- [ ] Debug data flow through CoopRayTrainer.fit()

#### Week 2
- [ ] Full AC training run (2 nodes, 4 GPUs each)
- [ ] Monitor training metrics (grounder/actor rewards, KL)
- [ ] Tune hyperparameters if needed

#### Week 3-4
- [ ] vLLM multi-LoRA rollout (Stage 2 speedup)
- [ ] GUI-Odyssey dataset adaptation
