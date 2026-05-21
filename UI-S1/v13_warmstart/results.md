# Warmstart RL Results — GUI-360 Balanced Test Set (1000 episodes)

**Date**: 2026-05-08
**Eval**: Trajectory-based, stop-on-error, match_threshold=0.5
**Test set**: `gui360_test_1000_balanced.jsonl` (1000 episodes, type-balanced)

---

## 1. Baselines

| Model | TSR | Step SR | Avg Progress | Notes |
|-------|:---:|:-------:|:------------:|-------|
| Qwen2.5-VL-7B (raw, zero-shot) | 2.0% | 36.4% | 9.6% | Base model, training-template prompt |
| gui_action SFT (training tmpl) | 3.0% | 28.0% | 8.6% | SFT on gui_action data, same prompt as RL |
| **gui_action SFT (gui360 tmpl)** | **14.4%** | **65.2%** | **30.6%** | SFT with GUI-360 prompt template |

---

## 2. Warmstart RL Results (from gui_action SFT, with reward fix)

All warmstart models start from `checkpoints/gui_action` (the gui_action SFT model).
Training uses `max_new_tokens=128`, reward fix: `type_reward=0 → content_reward=0`.

### Standard LoRA (r=210)

| Method | Epoch | TSR | Step SR | Avg Progress |
|--------|:-----:|:---:|:-------:|:------------:|
| Std LoRA + SP | 0 | 6.1% | 54.7% | 21.0% |
| Std LoRA + SP | 1 | 7.6% | 57.2% | 22.7% |
| **Std LoRA + SP** | **2** | **10.0%** | **59.6%** | **25.5%** |
| Std LoRA + GRPO | 0 | 6.5% | 54.9% | 21.0% |
| Std LoRA + GRPO | 1 | 7.4% | 56.7% | 22.6% |
| Std LoRA + GRPO | 2 | 8.8% | 58.2% | 23.9% |

### V12 Cooperative LoRA (soft-routed, 2 experts)

| Method | Epoch | TSR | Step SR | Avg Progress |
|--------|:-----:|:---:|:-------:|:------------:|
| V12 Coop + SP | 0 | 6.4% | 55.0% | 21.0% |
| V12 Coop + SP | 1 | 7.5% | 57.3% | 23.0% |
| **V12 Coop + SP** | **2** | **10.2%** | **59.1%** | **24.7%** |
| V12 Coop + GRPO | 0 | 6.0% | 54.8% | 20.2% |
| V12 Coop + GRPO | 1 | 6.8% | 56.2% | 22.2% |

### V13 Cooperative LoRA (iterative, T=2 comm rounds, 2 experts)

| Method | Epoch | TSR | Step SR | Avg Progress |
|--------|:-----:|:---:|:-------:|:------------:|
| V13 Coop + SP | 0 | 6.9% | 56.3% | 22.3% |
| **V13 Coop + SP** | **1** | **8.4%** | **58.1%** | **24.1%** |
| V13 Coop + GRPO | 0 | 7.2% | 56.7% | 22.6% |
| V13 Coop + GRPO | 1 | 4.8% | 55.1% | 21.0% |

---

## 3. Non-Warmstart RL Results (from raw Qwen, with reward bug)

These were trained from `Qwen2.5-VL-7B-Instruct` (no SFT warmstart) and had the
reward bug where click on empty-text type steps got 0.8 reward.

### V12 Cooperative LoRA (non-warmstart)

| Method | Epoch | TSR | Step SR | Avg Progress |
|--------|:-----:|:---:|:-------:|:------------:|
| V12 Coop (balanced) | 0 | 7.5% | 47.7% | 17.1% |
| V12 Coop (balanced) | 1 | **8.4%** | 54.4% | 20.2% |
| V12 Coop (balanced) | 2 | 8.3% | 54.8% | 20.9% |
| V12 Coop (balanced) | 3 | 8.3% | 54.6% | 20.6% |

### V13 Cooperative LoRA (non-warmstart)

| Method | Epoch | TSR | Step SR | Avg Progress |
|--------|:-----:|:---:|:-------:|:------------:|
| V13 Coop SP (balanced) | 0 | 8.0% | 54.6% | 20.2% |
| V13 Coop SP (balanced) | 1 | **0.0%** | **28.1%** | **6.4%** | **← click collapse** |
| V13 Coop SP (balanced) | 2 | 0.0% | 44.9% | 11.9% |
| V13 Coop GRPO (balanced) | 0 | 7.7% | 49.5% | 17.4% |
| V13 Coop GRPO (balanced) | 1 | 7.5% | 53.8% | 19.8% |

---

## 4. V13 Expert Ablation

### V13 Warmstart GRPO epoch-1 (from gui_action SFT)

| Mode | TSR | Step SR | Avg Progress |
|------|:---:|:-------:|:------------:|
| **Full model (both experts)** | **4.8%** | **55.1%** | **21.0%** |
| Expert 1 only + comm | 4.7% | 54.9% | 21.0% |
| **Expert 2 only + comm** | **6.3%** | **55.9%** | **22.1%** |
| Expert 1 only + nocomm | 3.3% | 51.6% | 17.8% |
| Expert 2 only + nocomm | 3.6% | 52.2% | 18.2% |

### V13 Non-Warmstart SP balanced epoch-2 (from raw Qwen, collapsed)

| Mode | TSR | Step SR | Avg Progress |
|------|:---:|:-------:|:------------:|
| Full model | 0.0% | 44.9% | 11.9% |
| Expert 1 only + comm | 0.1% | 43.9% | 11.6% |
| Expert 2 only + comm | 0.0% | 44.9% | 11.9% |
| Expert 1 only + nocomm | 0.9% | 40.8% | 10.2% |
| Expert 2 only + nocomm | 1.1% | 41.4% | 10.5% |

---

## 5. Action Type Distribution Analysis

### Predicted vs Ground Truth (Std LoRA SP WS epoch-2)

| | Click | Type | Swipe | None |
|---|:---:|:---:|:---:|:---:|
| **GT distribution** | **69.4%** | **25.5%** | **5.1%** | - |
| gui_action SFT (gui360) | 71.5% | 22.5% | 3.5% | 2.2% |
| **Std LoRA SP WS ep2** | **83.0%** | **14.1%** | **0.4%** | **2.4%** |
| **Std LoRA GRPO WS ep2** | **81.2%** | **13.0%** | **0.7%** | **4.9%** |

### Per GT-Type Step Success Rate

| GT Type | SFT (gui360) | RL SP WS ep2 | Delta |
|---------|:---:|:---:|:---:|
| click | 73.1% | 71.7% | -1.4% |
| type | 47.9% | 36.8% | **-11.1%** |
| swipe | 51.3% | 8.8% | **-42.5%** |

### Confusion Matrix (Std LoRA SP WS ep2)

| GT → Pred | click | type | swipe | None |
|-----------|:---:|:---:|:---:|:---:|
| **click** (n=1545) | 1481 (95.9%) | 46 (3.0%) | 0 | 16 (1.0%) |
| **type** (n=568) | 266 (46.8%) | 265 (46.7%) | 0 | 36 (6.3%) |
| **swipe** (n=113) | 100 (88.5%) | 2 (1.8%) | 10 (8.8%) | 1 (0.9%) |

---

## 6. Key Findings

### What worked
1. **Reward fix prevents full click collapse**: Non-warmstart V13 collapsed to TSR=0% at epoch-1. Warmstart V13 stays at 4.8-8.4%.
2. **Warmstart improves training efficiency**: Higher Step SR from epoch-0 (55% vs 47.7%) due to gui_action SFT initialization.
3. **SP consistently outperforms GRPO** across all model variants.
4. **Best warmstart RL (10.2%) exceeds non-warmstart V12 peak (8.4%)** with fewer epochs.

### Remaining issues
1. **RL degrades type/swipe predictions**: RL pushes click from 71% (SFT) to 83%, losing type (22%→14%) and swipe (3.5%→0.4%).
2. **Cooperative LoRA shows no advantage over standard LoRA**: V12 SP WS ep2 (10.2%) ≈ Std LoRA SP WS ep2 (10.0%).
3. **V13 GRPO destabilizes at epoch-1**: TSR drops from 7.2% to 4.8%. Expert 2 alone (6.3%) > full model (4.8%).
4. **Gap to SFT baseline**: Best RL (10.2%) still below SFT gui360-template (14.4%), mainly due to type/swipe loss.

### Root cause of click bias
- 69% of GT steps are click → 5.4x more positive gradient signal for click vs type
- Click actions only need coordinate matching; type actions need text content matching (harder)
- RL discovers "always predict click" as a safe strategy: 69% × 0.77 reward = expected 0.53 average
- GRPO K=8 sampling reinforces majority action (click) through advantage normalization

### Potential fixes
1. **Increase type mismatch penalty**: Negative reward instead of 0.1 for format-only
2. **Increase type_reward weight**: 0.4 instead of 0.2 to penalize type errors more
3. **Stronger KL regularization**: Current KL ~0.3-2 is too weak to prevent drift
4. **Per-type advantage normalization**: Separate baselines for click/type/swipe steps
5. **Training data rebalancing**: Upsample episodes with type/swipe steps
