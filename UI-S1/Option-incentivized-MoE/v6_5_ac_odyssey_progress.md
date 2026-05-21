# Cooperative LoRA v6.5 → AC / Odyssey Migration Progress

**Goal.** Reproduce v6.5's GUI-360 SOTA recipe (50.06% on GUI-360-test) on
two mobile benchmarks — AndroidControl (AC) and GUI-Odyssey — as **two
independent training runs** (no joint training, no checkpoint reuse), then
compare against a **vanilla LoRA SFT baseline** trained on the same data
without the cooperative communication wrapper.

**Status.** All training scripts, evaluators, data prep, and the SFT
baseline pipeline are ready to launch. No jobs have been submitted yet.

---

## 1. Architecture recap (cooperative LoRA v6.5)

- 2-agent cooperative LoRA (`num_agents=2`, `cooperative_comm=True`)
- Token-level routing: image tokens → LoRA_V, text/action tokens → LoRA_A
  (`IMAGE_PAD_ID = 151655`, prompt-agnostic via `forward_pre_hook`)
- Per-layer cross-agent communication via tanh gate
  (`gate_type=tanh`, `gate_init=0.0`, `gate_lr_multiplier=100.0`)
- LoRA shape: `r=256, alpha=512, dropout=0.05`, target = all 7 modules
  (q/k/v/o/gate_proj/up_proj/down_proj)
- Loss = thought-CE only (`bind_weight=0`); thought ≈ visual description
- Hyperparams: `lr=1e-5`, `eff_bs=128`, `num_epochs=4`, `warmup=0.03`

**Why vLLM is incompatible.** vLLM's optimised forward path bypasses the
`forward_pre_hook` we use to route image vs text tokens through different
LoRA branches, so all generation **must** use HF `generate()`. This is the
reason we built dedicated HF trajectory evaluators rather than reusing the
existing vLLM-based eval scripts.

---

## 2. Deliverables

### 2.1 Shared HF eval helper

| File | Purpose |
|---|---|
| `evaluation/cooperative_trajectory_common.py` | `load_cooperative_model`, `cooperative_generate`, `safe_parse_response`, `shard_episodes`, `length_bucket`, `compute_trajectory_metrics`, `aggregate_action_stats`, `_json_default` |

This module is imported by both trajectory evaluators below so the routing
contract (vision tokens → branch V, text → branch A) is implemented exactly
once.

### 2.2 AC HF trajectory evaluator (M1)

| File | Notes |
|---|---|
| `evaluation/eval_cooperative_ac_trajectory.py` | JsonFormat + RAW_SPACE + cooperative HF generate |
| `scripts/exp_cooperative/eval_v6_5_ac_trajectory.slurm` | 4 shards × 4 GPUs, single node, 12h, action-type aggregation |

**Sharding.** Episodes are deterministically split round-robin across 4
shards; within a shard, GPUs work sequentially (each shard pins one GPU).
Per-shard JSON results are aggregated by the trailing wrap-up step in the
slurm file.

**CLI args:** `--base_model`, `--coop_checkpoint`, `--jsonl_file`,
`--output_dir`, `--gpu_id`, `--shard_id`, `--num_shards`,
`--n_history_image_limit`, `--max_new_tokens`, `--no_stop`.

### 2.3 Odyssey HF trajectory evaluator (M2)

| File | Notes |
|---|---|
| `evaluation/eval_cooperative_odyssey_trajectory.py` | Adds `pred_coord_to_1k`, per-category and per-device breakdowns |
| `scripts/exp_cooperative/eval_v6_5_odyssey_trajectory.slurm` | 4 shards × 4 GPUs, single node, category/device aggregation |

Per-step results capture `pred_coord_1k` and `gt_coord_1k`, mirroring the
existing `gui_odyssey_eval` scoring contract.

### 2.4 Thought-augmented data (M3)

#### AC (M3a) — `datasets/cooperative_thought_ac/`

| File | Bytes | Lines |
|---|---|---|
| `prepare_ac_thought.py` | 10 257 | — |
| `ac_train_thought.jsonl` | 27.5 MB | **6 482** |
| `ac_val_thought.jsonl` | 224 KB | **54** |

- Source: `datasets/ui_s1_dataset/ui_s1_train_with_desc.jsonl`
  (1 000 episodes, `desc_t1` per step)
- Train/val split: episode-level, 50 episodes held out, seed 42
- 84.7% of train samples carry a `<think>` segment; 63.8% carry `gt_coords`
- Action distribution: click 3 351 · terminate 990 · swipe 777 · type 386 ·
  open 374 · wait 370 · system_button 226 · long_press 8

#### Odyssey (M3b) — `datasets/cooperative_thought_odyssey/`

| File | Bytes | Lines |
|---|---|---|
| `prepare_odyssey_thought.py` | 12 184 | — |
| `odyssey_train_thought.jsonl` | 385 MB | **101 364** |
| `odyssey_val_thought.jsonl` | 2.6 MB | **722** |

- Source: `datasets/GUI-Odyssey/splits/random_split.json` (6 668 episodes)
  + per-episode `annotations/{episode_id}.json`
- Reuses `gui_odyssey_eval/convert_to_eval_format.py:convert_action`
- Thought source = `description` (visual scene description), **not**
  `intention` (action rationale) — visual descriptions align better with
  the cooperative binding signal
- Train/val split: episode-level, 200 episodes held out, seed 42
- 100% of train samples carry a thought (Odyssey annotates every step)

**Format (both datasets).** LLaMA-Factory ShareGPT-style, single image:

```json
{
  "conversations": [
    {"from": "human",     "value": "<image>\n<system>\n## User Instruction\n{goal}\n## History of previous actions\nStep 1: <action>{...}</action>\n..."},
    {"from": "assistant", "value": "<think>{description}</think>\n<action>{...}</action>"}
  ],
  "images": ["/abs/screenshot.png"],
  "has_thought": true,
  "gt_coords": [x, y]
}
```

`train_cooperative.py` consumes only `images[0]`, so multi-step history is
flattened into the user text as `Step N: <action>{...}</action>` lines.

### 2.5 Cooperative training (M4)

| File | Nodes × GPU | grad_accum | eff_bs | Output dir |
|---|---|---|---|---|
| `scripts/exp_cooperative/train_v6_5_ac_comm_thought.slurm`      | 2 × 4 | 16 | 128 | `train_GUI_360/llamafactory/output/cooperative_v6_5_ac` |
| `scripts/exp_cooperative/train_v6_5_odyssey_comm_thought.slurm` | 8 × 4 |  4 | 128 | `train_GUI_360/llamafactory/output/cooperative_v6_5_odyssey` |

Both runs share the v6.5 hyperparameter wall: tanh gate, `gate_init=0`,
`gate_lr_mult=100`, `lr=1e-5`, 4 epochs, frozen vision tower + projector,
LoRA r=256/α=512, all 7 modules. AC uses fewer nodes because the dataset
is ~15× smaller; Odyssey matches the original GUI-360 v6.5 layout because
sample counts are within ~3% (101 K vs 98 K).

### 2.6 Vanilla SFT baseline (NEW)

The baseline has to be a clean A/B partner for cooperative v6.5: same
training data, same LoRA shape, **no** communication wrapper. We use
LLaMA-Factory's stock LoRA SFT pipeline because (a) it requires zero
custom code, and (b) the resulting LoRA adapter merges cleanly into the
base model so it can be evaluated with the existing **vLLM** trajectory
scripts (`scripts/eval/ac/eval_a_ar_trajectory.py`,
`gui_odyssey_eval/eval_ar_trajectory.py`) — no HF generate() workaround
needed.

**Dataset registration** — `train_GUI_360/llamafactory/data/dataset_info.json`
gained four new entries:

```
ac_thought_train      -> datasets/cooperative_thought_ac/ac_train_thought.jsonl
ac_thought_val        -> datasets/cooperative_thought_ac/ac_val_thought.jsonl
odyssey_thought_train -> datasets/cooperative_thought_odyssey/odyssey_train_thought.jsonl
odyssey_thought_val   -> datasets/cooperative_thought_odyssey/odyssey_val_thought.jsonl
```

All four use ShareGPT format with `assistant_tag: "assistant"` (cooperative
data uses `assistant`, while existing GUI-360/AC entries use `gpt`).
LLaMA-Factory accepts `.jsonl` natively (`FILEEXT2TYPE["jsonl"] = "json"`),
and absolute paths in `file_name` work because `os.path.join` collapses
them.

**LoRA SFT configs:**

| File | Dataset | grad_accum |
|---|---|---|
| `train_GUI_360/llamafactory/qwen25vl_v6_5_ac_sft_baseline.yaml`      | `ac_thought_train` / `_val`      | 16 |
| `train_GUI_360/llamafactory/qwen25vl_v6_5_odyssey_sft_baseline.yaml` | `odyssey_thought_train` / `_val` |  4 |

Hyperparameters intentionally mirror cooperative v6.5:

| Knob | v6.5 cooperative | SFT baseline |
|---|---|---|
| LoRA rank | 256 | 256 |
| LoRA alpha | 512 | 512 |
| LoRA dropout | 0.05 | 0.05 |
| Target modules | q/k/v/o/gate/up/down | q/k/v/o/gate/up/down |
| Vision tower | frozen | frozen |
| Projector | frozen (whole base frozen) | frozen |
| Optimizer LR | 1e-5 | 1e-5 |
| Epochs | 4 | 4 |
| Effective batch | 128 | 128 |
| Warmup | 0.03 | 0.03 |
| Weight decay | 0.0 | 0.0 |
| LR schedule | cosine | cosine |
| Cooperative wrapper | **YES** (tanh gate, lr_mult=100) | **NO** |
| Cross-agent communication | **YES** | **NO** |
| Number of LoRAs | 2 (per-token routing) | 1 |

The remaining delta is exactly what we want to measure: the value of the
cooperative wrapper / per-layer tanh communication on these two corpora.

**Slurm scripts:**

| File | Nodes × GPU | Time | Pairs with |
|---|---|---|---|
| `scripts/exp_cooperative/train_v6_5_ac_sft_baseline.slurm`      | 2 × 4 |  6 h | `train_v6_5_ac_comm_thought.slurm` |
| `scripts/exp_cooperative/train_v6_5_odyssey_sft_baseline.slurm` | 8 × 4 | 24 h | `train_v6_5_odyssey_comm_thought.slurm` |

Both launch via `llamafactory-cli train <yaml>` under `srun`, with the
same node-local `HF_DATASETS_CACHE` trick used by the existing GUI-360 SFT
slurm files to avoid NFS mmap SIGBUS errors.

---

## 3. End-to-end run order

For each benchmark (AC, Odyssey) the workflow is:

1. **Data prep** — already done, JSONL files present on disk
   (no need to rerun unless you change the prompt template).
2. **Cooperative train** — `sbatch scripts/exp_cooperative/train_v6_5_{ac,odyssey}_comm_thought.slurm`
3. **Vanilla SFT baseline train** — `sbatch scripts/exp_cooperative/train_v6_5_{ac,odyssey}_sft_baseline.slurm`
4. **Eval cooperative checkpoint** —
   `sbatch scripts/exp_cooperative/eval_v6_5_{ac,odyssey}_trajectory.slurm`
   (HF generate, sharded; reads `cooperative_v6_5_{ac,odyssey}/` adapter)
5. **Eval SFT baseline** — merge LoRA into base via the existing
   `evaluation/merge_cooperative_lora.py` style helper (or LLaMA-Factory
   `export` command), then run the **vLLM**-based AC / Odyssey trajectory
   evaluators (`scripts/eval/ac/eval_a_ar_trajectory.py`,
   `gui_odyssey_eval/eval_ar_trajectory.py`) — these are fast and already
   sharded.
6. **Compare** — both pairs of numbers land in the same metric format
   (action accuracy, type accuracy, grounding accuracy, per-action breakdown).
   Direct delta = "value of cooperative wrapper".

Steps 2 and 3 are independent and can be queued in parallel.

---

## 4. Open items / things to verify before submission

- [ ] Confirm LLaMA-Factory tolerates the absolute `file_name` paths added
  to `dataset_info.json` (it should — the loader does `os.path.join` which
  collapses absolute right-hand sides). Watch the first few preprocess
  steps of the baseline run.
- [ ] Confirm `freeze_multi_modal_projector: true` is the right knob for
  Qwen2.5-VL in this LLaMA-Factory revision. If the projector is still
  unfrozen, the baseline gets a small unfair advantage.
- [ ] After cooperative v6.5-AC finishes, run
  `evaluation/eval_cooperative_ac_trajectory.py` on a 10-episode smoke
  shard before launching the full 4-shard sweep, to catch any wrapper
  loading regression.
- [ ] Same smoke test for Odyssey before the full shard sweep.
- [ ] LLaMA-Factory `export` step is needed to materialise the SFT
  baseline LoRA into a vLLM-loadable adapter. Pre-stage this command in
  the eval slurm so it runs automatically after training.

---

## 5. File index (everything created for this milestone)

```
evaluation/
  cooperative_trajectory_common.py            # shared HF helper
  eval_cooperative_ac_trajectory.py           # M1: AC HF eval
  eval_cooperative_odyssey_trajectory.py      # M2: Odyssey HF eval

datasets/
  cooperative_thought_ac/
    prepare_ac_thought.py                     # M3a
    ac_train_thought.jsonl                    # 6 482 samples
    ac_val_thought.jsonl                      # 54 samples
  cooperative_thought_odyssey/
    prepare_odyssey_thought.py                # M3b
    odyssey_train_thought.jsonl               # 101 364 samples
    odyssey_val_thought.jsonl                 # 722 samples

scripts/exp_cooperative/
  train_v6_5_ac_comm_thought.slurm            # M4a cooperative
  train_v6_5_odyssey_comm_thought.slurm       # M4b cooperative
  eval_v6_5_ac_trajectory.slurm               # M1 launcher
  eval_v6_5_odyssey_trajectory.slurm          # M2 launcher
  train_v6_5_ac_sft_baseline.slurm            # baseline launcher
  train_v6_5_odyssey_sft_baseline.slurm       # baseline launcher

train_GUI_360/llamafactory/
  qwen25vl_v6_5_ac_sft_baseline.yaml          # baseline config
  qwen25vl_v6_5_odyssey_sft_baseline.yaml     # baseline config
  data/dataset_info.json                      # +4 dataset entries
```
