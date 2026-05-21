# v10: Sequential Cooperative LoRA with Natural Language Communication

## 1. Motivation

### Why v6.5/v9 failed
- v6.5 ablation: `t_only` (all→LoRA_A) **outperforms** `hard` (image→V, text→A): 10.33% vs 10.11% TSR
- LoRA_V processing image tokens is a **net negative** — it interferes with the base model's visual understanding
- v9's credit-based reweighted-SFT didn't help (9.79% best vs 10.31% v6.5) because the problem is **architectural**, not in training signal
- Latent communication gates (W_av, W_va) carry no interpretable information

### v10 core insight
Instead of routing tokens through different LoRAs in one forward pass, run **two sequential generation passes**:
1. **Grounder (LoRA_V)**: generates natural language grounding description
2. **Actor (LoRA_A)**: reads grounder's description as input, generates action

Communication is **explicit natural language**, not latent vectors.

---

## 2. Architecture

### 2.1 Sequential Two-Pass Generation

```
┌─────────────────────────────────────────────────────────┐
│ Input: screenshot + goal + action_history               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Pass 1: Grounder (base_model + LoRA_V)                  │
│                                                         │
│ Prompt: "Determine the action type and describe target." │
│                                                         │
│ Output: <action_type>click</action_type>                │
│         <target>The blue 'Submit' button at bottom-right │
│         of the form</target>                            │
└────────────────────┬────────────────────────────────────┘
                     │ (action_type + target description)
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Pass 2: Actor (base_model + LoRA_A)                     │
│                                                         │
│ Prompt: original input + action type + target           │
│                                                         │
│ Output: {"action":"click","coordinate":[920,1800]}      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Key Differences from v6.5/v9

| Aspect | v6.5/v9 | v10 |
|--------|---------|-----|
| Forward passes | 1 (per-token routing) | 2 (sequential) |
| Communication | Latent gates W_av/W_va | Natural language text |
| LoRA_V role | Process image tokens | Generate grounding description |
| LoRA_A role | Generate action | Read grounding + generate action |
| Image tokens | Through LoRA_V | Through base model (no LoRA) |
| Cooperative wrapper | Required (token routing) | Not needed |
| Interpretability | Black box | Grounding text is human-readable |

### 2.3 LoRA Configuration

Each LoRA is a **standard PEFT LoRA adapter** (no cooperative wrapper needed):

```python
# Grounder LoRA (V)
lora_v_config = LoraConfig(
    r=256, lora_alpha=512, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

# Actor LoRA (A)
lora_a_config = LoraConfig(
    r=256, lora_alpha=512, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)
```

---

## 3. Prompt Design

### 3.1 Grounder Prompt (Updated: action type + structured output)

The grounder now determines the **action type** in addition to describing the target, using a structured `<action_type>` + `<target>` output format. This addresses the action type collapse problem (see `eval_s300_analysis.md`).

```
<|im_start|>system
You are a GUI grounding agent. Given a screenshot and an instruction,
determine the next action type and describe the target.

Output format:
<action_type>one of: click, type, open, swipe, long_press, wait, system_button, terminate</action_type>
<target>description of the target (UI element location for click/long_press,
app name for open, scroll direction for swipe, button name for system_button,
reason for wait, or text to type)</target><|im_end|>
<|im_start|>user
<image>
Instruction: {goal}
History: {action_history}
Determine the action type and describe the target.<|im_end|>
<|im_start|>assistant
<action_type>click</action_type>
<target>The blue 'Submit' button located in the bottom-right area of the form</target>
<|im_end|>
```

**Grounder output parsing** (`parse_grounder_output`): Extracts `(action_type, target)` from the structured tags. Falls back to `("unknown", full_text)` if parsing fails, ensuring backward compatibility.

### 3.2 Actor Prompt (Updated: receives action type + target separately)

The actor now receives the grounder's action type and target description as **separate fields**, giving it an explicit type signal.

```
<|im_start|>system
You are a GUI agent. Given a screenshot, instruction, and grounding
analysis (action type + target description), perform the next action.
Output format: <action>{"action": "...", ...}</action><|im_end|>
<|im_start|>user
<image>
Instruction: {goal}
History: {action_history}

Grounding action type: click
Grounding target: The blue 'Submit' button located in the bottom-right area of the form

Output the next action.<|im_end|>
<|im_start|>assistant
<action>{"action":"click","coordinate":[920,1800]}</action>
<|im_end|>
```

---

## 4. Training: GRPO with Dual Rewards

### 4.1 Overview

```
For each training sample (screenshot, goal, history, gt_action):
  1. Grounder samples K descriptions: g_1, g_2, ..., g_K
  2. For each g_k, Actor samples M actions: a_k1, a_k2, ..., a_kM
  3. Evaluate rewards:
     - r_coord(a_km) = coord_correct(predicted_coord, gt_coord)
     - r_action(a_km) = action_correct(predicted_action, gt_action)
  4. Compute advantages:
     - A_grounder(g_k) = normalized mean reward across actor samples for g_k
     - A_actor(a_km) = normalized reward within group for g_k
  5. Update:
     - LoRA_V: GRPO loss on grounder tokens weighted by A_grounder
     - LoRA_A: GRPO loss on actor tokens weighted by A_actor
```

### 4.2 Reward Functions

#### Grounder Reward: `coord_correct`
```python
def grounder_reward(predicted_action, gt_action, threshold=0.05):
    """
    Binary reward: is the predicted coordinate within threshold of GT?

    For click actions: compare normalized coordinates
    For non-click actions (scroll, type, open): reward = 0.5 (neutral)
    """
    if gt_action["action"] not in ("click", "long_press"):
        return 0.5  # neutral for non-coordinate actions

    pred_coord = predicted_action.get("coordinate")
    gt_coord = gt_action["coordinate"]
    if pred_coord is None:
        return 0.0

    # Normalize to [0, 1]
    dx = abs(pred_coord[0] - gt_coord[0]) / image_width
    dy = abs(pred_coord[1] - gt_coord[1]) / image_height
    dist = (dx**2 + dy**2) ** 0.5

    if dist < threshold:
        return 1.0
    elif dist < threshold * 2:
        return 0.5  # partial credit
    else:
        return 0.0
```

#### Actor Reward: `action_correct`
```python
def actor_reward(predicted_action, gt_action, image_w, image_h):
    """
    Existing action correctness evaluation.
    Checks: action_type match + content match (coordinate/text/direction).
    Uses evaluate_android_control_action() from existing codebase.
    """
    type_match, extract_match = evaluate_android_control_action(
        predicted_action, gt_action, image_w, image_h
    )
    if extract_match:
        return 1.0
    elif type_match:
        return 0.3  # partial credit for correct action type
    else:
        return 0.0
```

### 4.3 Advantage Computation

**Grounder advantage** — aggregate across actor samples:
```python
# For each grounder output g_k, average the actor rewards
grounder_reward_k = mean([r_coord(a_km) for m in range(M)])

# GRPO normalization across K grounder samples
A_grounder = (grounder_rewards - mean(grounder_rewards)) / (std(grounder_rewards) + eps)
```

**Actor advantage** — standard GRPO within each grounder group:
```python
# For each grounder output g_k, normalize actor rewards across M samples
A_actor_km = (r_action(a_km) - mean(r_action(a_k*))) / (std(r_action(a_k*)) + eps)
```

### 4.4 Loss Computation

```python
# Grounder loss (update LoRA_V)
loss_grounder = -sum(A_grounder[k] * log_prob_grounder(g_k)) / K

# Actor loss (update LoRA_A)
loss_actor = -sum(A_actor[k,m] * log_prob_actor(a_km)) / (K * M)

# KL regularization (against v6.5 reference policy)
kl_grounder = KL(pi_v || pi_ref)
kl_actor = KL(pi_a || pi_ref)

# Total loss
loss = loss_grounder + loss_actor + beta * (kl_grounder + kl_actor)
```

### 4.5 Simplified Variant (Recommended for Step 1)

Start with **K=4 grounder samples, M=1 actor sample** per grounder output:

```
For each sample:
  1. Grounder samples 4 descriptions: g_1..g_4
  2. Actor generates 1 action per description: a_1..a_4
  3. Grounder reward: r_g(k) = coord_correct(a_k)
  4. Actor reward: r_a(k) = action_correct(a_k)
  5. GRPO advantage (shared group of 4):
     - A_g(k) for grounder tokens
     - A_a(k) for actor tokens
  6. Joint update
```

This reduces compute by 4x vs K=4,M=4.

---

## 5. Inference Pipeline

### 5.1 Two-Pass Inference

```python
def v10_inference(image, goal, history, grounder_model, actor_model):
    """
    Sequential two-pass inference.

    grounder_model = base_model merged with LoRA_V
    actor_model = base_model merged with LoRA_A
    """
    # Pass 1: Grounder
    grounder_prompt = format_grounder_prompt(image, goal, history)
    grounding_text = grounder_model.generate(grounder_prompt)

    # Pass 2: Actor
    actor_prompt = format_actor_prompt(image, goal, history, grounding_text)
    action_text = actor_model.generate(actor_prompt)

    return parse_action(action_text)
```

### 5.2 vLLM Deployment

Two options:

**Option A: Two vLLM instances (recommended for eval)**
```bash
# Merge LoRA_V into base → grounder_model
# Merge LoRA_A into base → actor_model
# Start two vLLM servers on different ports
python -m vllm.entrypoints.openai.api_server --model grounder_model --port 8000
python -m vllm.entrypoints.openai.api_server --model actor_model --port 8001
```

**Option B: Single vLLM with LoRA switching**
```bash
# Use vLLM's multi-LoRA support
# Load base model + both LoRA adapters
# Switch adapter per request
python -m vllm.entrypoints.openai.api_server \
    --model base_model \
    --enable-lora \
    --lora-modules grounder=path/to/lora_v actor=path/to/lora_a
```

**Option C: Single merged model (eval speed priority)**
```bash
# For eval: merge both LoRAs into base model
# Grounder pass: use merged_v model
# Actor pass: use merged_a model
# Requires 2x model memory or sequential loading
```

### 5.3 AR Trajectory Eval

```python
for episode in eval_episodes:
    state = initial_state(episode)
    for step in range(max_steps):
        # Two-pass generation
        grounding = grounder.generate(state.screenshot, state.goal, state.history)
        action = actor.generate(state.screenshot, state.goal, state.history, grounding)

        # Evaluate and advance
        correct = evaluate_action(action, state.gt_action)
        if not correct:
            break
        state = advance(state, action)
```

---

## 6. Warm-Start Strategy

### 6.1 From v6.5 ep4 Checkpoint

v6.5 checkpoint has: `lora_v.pt`, `lora_a.pt`, `lora_comm.pt`

**For LoRA_V (Grounder):**
- v6.5's LoRA_V was trained on image tokens → not directly useful for text generation
- **Option A**: Initialize LoRA_V from LoRA_A weights (since A already knows how to generate text)
- **Option B**: Initialize LoRA_V from scratch (random init like standard LoRA)
- **Option C**: Initialize LoRA_V from LoRA_A, then do a few SFT warmup steps on grounding data
- **Recommended**: Option A — copy LoRA_A weights to LoRA_V, so both start with text generation capability

**For LoRA_A (Actor):**
- v6.5's LoRA_A was trained to generate `<think>...<action>` → good starting point
- Direct warm-start from v6.5 LoRA_A
- The actor already knows the action format, just needs to learn to use grounder's input

### 6.2 Weight Extraction

```python
# Extract LoRA_A and LoRA_V from cooperative checkpoint
coop_ckpt = "train_GUI_360/llamafactory/output/cooperative_v6_5_ac/epoch-4"

lora_a = torch.load(f"{coop_ckpt}/lora_a.pt")  # Actor weights
lora_v = torch.load(f"{coop_ckpt}/lora_v.pt")  # Vision weights (not used)

# For v10 grounder: initialize from LoRA_A
# Rename lora_A_a/lora_B_a → standard PEFT naming
grounder_init = convert_cooperative_to_peft(lora_a, suffix="a")

# For v10 actor: use LoRA_A directly
actor_init = convert_cooperative_to_peft(lora_a, suffix="a")
```

---

## 7. Implementation Plan

### 7.1 Files to Create

```
v10/
├── v10_design.md              # This document
├── reward.py                  # Grounder + Actor reward functions
├── train_grpo.py              # GRPO trainer with sequential generation
├── convert_checkpoint.py      # Convert v6.5 cooperative ckpt → 2x PEFT LoRA
├── inference.py               # Two-pass inference logic
├── eval_trajectory.py         # AR trajectory eval with two-pass
├── scripts/
│   ├── train_v10_grpo.slurm   # Training launch script
│   ├── eval_v10_trajectory.slurm  # Eval launch script
│   └── logs/
```

### 7.2 Existing Infrastructure to Reuse

| Component | Source | Usage |
|-----------|--------|-------|
| GRPO advantage | `verl/trainer/ppo/core_algos.py` | `compute_grpo_outcome_advantage()` |
| Policy loss | `verl/trainer/ppo/core_algos.py` | `compute_policy_loss()` with clipping |
| KL penalty | `verl/trainer/ppo/core_algos.py` | `kl_penalty()` |
| GUI-360 reward | `verl/utils/reward_score/gui360/reward.py` | Soft coordinate scoring |
| Action eval | `evaluation/cooperative_trajectory_common.py` | `evaluate_android_control_action()` |
| vLLM rollout | `train/srun_grpo/scripts/train_srun_grpo_vllm.py` | vLLM generation integration |
| Srun GRPO trainer | `train/train_srun_grpo_worker.py` | Base trainer structure |

### 7.3 Implementation Steps

**Step 1: Checkpoint conversion** (`convert_checkpoint.py`)
- Convert v6.5 cooperative checkpoint → two separate PEFT LoRA checkpoints
- Grounder: init from LoRA_A weights (text generation capability)
- Actor: init from LoRA_A weights (action generation capability)

**Step 2: Reward functions** (`reward.py`)
- `grounder_reward()`: coord_correct with soft scoring
- `actor_reward()`: action_correct using existing evaluation
- Integration with verl reward manager interface

**Step 3: Sequential generation** (`inference.py`)
- `SequentialCooperativeAgent` class
- `generate_grounding(image, goal, history) → str`
- `generate_action(image, goal, history, grounding) → dict`
- Prompt formatting for both passes

**Step 4: GRPO trainer** (`train_grpo.py`)
- Extend `SrunGRPOTrainer` from existing infrastructure
- Sequential rollout: grounder → actor
- Dual advantage computation
- Separate LoRA parameter groups for grounder/actor updates
- Reference policy KL computation

**Step 5: Eval script** (`eval_trajectory.py`)
- AR trajectory eval with two-pass generation
- Metrics: TSR, avg_progress, per-action accuracy

**Step 6: Slurm scripts** (`scripts/`)
- Training: multi-node GRPO with vLLM rollouts
- Eval: single-node trajectory evaluation

---

## 8. Training Configuration

```yaml
# v10 GRPO config
model:
  base: checkpoints/Qwen2.5-VL-7B-Instruct
  grounder_lora: v10/checkpoints/grounder_init/  # from convert_checkpoint.py
  actor_lora: v10/checkpoints/actor_init/

training:
  algorithm: grpo
  num_grounder_samples: 4      # K
  num_actor_samples: 1         # M (start simple)
  temperature: 1.0
  top_p: 1.0
  max_new_tokens_grounder: 256
  max_new_tokens_actor: 128

  grounder:
    learning_rate: 1e-5
    kl_coef: 0.01
    clip_ratio: 0.2

  actor:
    learning_rate: 5e-6
    kl_coef: 0.01
    clip_ratio: 0.2

  batch_size: 8
  gradient_accumulation: 4
  num_epochs: 10
  warmup_ratio: 0.03

reward:
  grounder: coord_correct      # soft coordinate scoring
  actor: action_correct        # full action match
  coord_threshold: 0.05        # normalized distance threshold

data:
  train: datasets/cooperative_thought_ac/ac_train_thought.jsonl
  val: datasets/cooperative_thought_ac/ac_val_thought.jsonl
```

---

## 9. Expected Advantages

1. **Clean credit assignment**: Grounder and actor have independent reward signals
2. **Interpretable communication**: Grounding text is human-readable, debuggable
3. **No image→V routing harm**: LoRA_V never processes image tokens
4. **Modular**: Each LoRA can be independently evaluated, ablated, or replaced
5. **Leverage existing GRPO**: Reuse proven RL infrastructure from verl/
6. **Natural curriculum**: Grounder learns to produce descriptions that help actor succeed

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| 2x inference cost (two passes) | Grounder output is short (~50-100 tokens); vLLM batching amortizes overhead |
| Grounder may not learn useful descriptions from scratch | Warm-start from LoRA_A (already generates text); GRPO reward guides toward useful grounding |
| Credit assignment between grounder and actor | Separate reward signals; grounder reward = coord_correct is direct |
| KL divergence explosion early in training | Conservative KL coefficient (0.01); clip ratio (0.2) |
| Grounder outputs may be noisy/inconsistent | Temperature annealing; best-of-K selection at eval time |
| Non-click actions have no coord to evaluate grounder | Neutral reward (0.5) for non-coordinate actions; actor reward still provides signal |
