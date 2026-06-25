# V23 Offline Visual Transition Optimization

Goal: improve the current offline GUI-360 evaluation without online environment access.

The benchmark uses GT screenshots and `stop_on_error`, so V23 optimizes prefix survival on expert states rather than online recovery.

The consolidated OPD/self-distillation direction and full experiment history are
in [`opd_self_distillation.md`](opd_self_distillation.md).

## Current Tools

```text
prepare_offline_data.py          Build weighted action examples and GT transition examples.
mine_hard_states_from_eval.py    Mine first-error and pre-first-error states from eval_results JSON.
build_hard_replay.py             Convert mined hard states into weighted ShareGPT replay examples.
sample_matcher_candidates.py     Sample K action candidates on GT screens and score them with the existing matcher.
train_offline_grpo.py            Train from fixed matcher-scored candidates without online generation.
build_where_what_dataset.py      Decompose GUI actions into WHAT and WHERE expert supervision.
train_where_what_routed_sft.py   Train full tool calls with token-level WHAT/WHERE route supervision.
train_route_pairwise.py          Route-only pairwise preference training on fixed matcher hard negatives.
analyze_comm_gate_information.py Diagnose WHAT/WHERE information carried by V13 comm gates.
```

## Generated Artifacts

```text
outputs/v23_visual_transition/gui360_test_offline_data/
  offline_action_examples.jsonl       # 7498 weighted action examples
  offline_transition_examples.jsonl   # 6498 adjacent-screen transition examples
  summary.json

datasets/gui360-balanced/gui360_train_from_parquet.jsonl
datasets/gui360-balanced/images/train/
  # converted from train parquet: 1573 episodes, 12574 screenshots

outputs/v23_visual_transition/gui360_train_offline_data/
  offline_action_examples.jsonl       # 12574 weighted action examples
  offline_transition_examples.jsonl   # 11001 adjacent-screen transition examples
  summary.json

outputs/v23_visual_transition/train_eval_full_sft_8gpu64_stop/
  eval_summary_20260625_043340.json   # full SFT on train split: 38.9 TSR / 83.5 StepSR
  eval_results_20260625_043340.json

outputs/v23_visual_transition/train_full_sft_hard_states/
  hard_states.jsonl                   # 1642 hard rows from train split full SFT eval
  hard_state_summary.json

outputs/v23_visual_transition/train_full_sft_hard_replay/
  hard_action_replay.jsonl            # 1642 weighted train hard replay examples
  hard_replay_summary.json

outputs/v23_visual_transition/train_full_sft_k8_candidates_smoke/
  matcher_candidates.jsonl            # 20-row K=4 smoke; 35% any-success rate
  summary.json

outputs/v23_visual_transition/train_full_sft_k8_candidates/
  matcher_candidates.jsonl            # target full K=8 candidate set for offline GRPO
  summary.json

checkpoints/v23_offline_candidate_grpo_full_sft/
  epoch-*_step-*/cooperative/          # offline candidate GRPO checkpoints

outputs/v23_visual_transition/where_what_train_dataset/
  what_examples.jsonl                  # 13,829 examples: function/status/non-location args
  where_examples.jsonl                 # 9,271 examples: coordinate/start/end only
  paired_where_what_examples.jsonl     # paired structured records for custom expert routing
  summary.json

outputs/v23_visual_transition/where_what_routed_sft_smoke/
checkpoints/v23_where_what_routed_sft_smoke/
  # 1-GPU smoke: route loss drops and r_what > r_where by step 2

checkpoints/v23_where_what_routed_sft_short/
  epoch-0_step-5/                    # first V23 full balanced improvement
  epoch-0_step-10/
  epoch-0_step-15/
  epoch-0_step-20/
  epoch-0_step-25/

outputs/v23_visual_transition/eval_where_what_routed_step5_balanced_vllm_1000_stop/
  eval_summary_20260625_095715.json  # 22.1 TSR / 68.49 StepSR on balanced 1000

checkpoints/v23_where_what_route_only_from_step5_short_fullsave/
  epoch-0_step-10/                    # current best V23 full balanced result

outputs/v23_visual_transition/eval_route_only_from_step5_step10_balanced_vllm_1000_stop/
  eval_summary_20260625_104147.json  # 22.3 TSR / 69.07 StepSR on balanced 1000

checkpoints/v23_where_what_where_heavy_from_routeonly_step10_short/
  epoch-0_step-10/                    # negative 200-slice result; do not full-eval yet

outputs/v23_visual_transition/eval_where_heavy_from_routeonly_step10_balanced_vllm_200_stop/
  eval_summary_*.json                 # 18.0 TSR / 66.67 StepSR on first 200

checkpoints/v23_where_what_route_only_from_step10_lowroute_short/
  epoch-0_step-10/                    # negative 200-slice result; continued route-only overfits/saturates

outputs/v23_visual_transition/eval_route_only_from_step10_lowroute_balanced_vllm_200_stop/
  eval_summary_*.json                 # 18.0 TSR / 66.94 StepSR on first 200

checkpoints/v23_route_pairwise_from_routed_step5_short/
  epoch-0_step-5/
  epoch-0_step-10/                    # neutral/negative 200-slice result

outputs/v23_visual_transition/eval_route_pairwise_step5_balanced_vllm_200_stop/
  eval_summary_*.json                 # 18.0 TSR / 67.20 StepSR on first 200

outputs/v23_visual_transition/eval_route_pairwise_step10_balanced_vllm_200_stop/
  eval_summary_*.json                 # 18.5 TSR / 66.73 StepSR on first 200

outputs/v23_visual_transition/comm_gate_info_current_best_128/
  comm_gate_information_report.md      # current-best comm gates are near-constant around 0.501

checkpoints/v23_comm_gate_only_from_best_short/
  epoch-0_step-5/
  epoch-0_step-10/                     # opens g21>g12 on WHERE, but negative 200-slice

outputs/v23_visual_transition/comm_gate_info_comm_gate_only_step10_128/
  comm_gate_information_report.md      # WHERE (g21-g12) rises from ~0.00004 to ~0.0203

outputs/v23_visual_transition/eval_comm_gate_step5_balanced_vllm_200_stop/
  eval_summary_*.json                  # 18.0 TSR / 66.53 StepSR on first 200

outputs/v23_visual_transition/eval_comm_gate_step10_balanced_vllm_200_stop/
  eval_summary_*.json                  # 18.5 TSR / 67.00 StepSR on first 200

checkpoints/v23_comm_gate_anchor_from_best_short/
  epoch-0_step-5/
  epoch-0_step-10/                     # behavior-anchored comm gate, still below current best

outputs/v23_visual_transition/eval_comm_gate_anchor_step5_balanced_vllm_200_stop/
  eval_summary_*.json                  # 17.5 TSR / 66.19 StepSR on first 200

outputs/v23_visual_transition/eval_comm_gate_anchor_step10_balanced_vllm_200_stop/
  eval_summary_*.json                  # 18.5 TSR / 67.20 StepSR on first 200

checkpoints/v23_comm21_anchor_from_best_short/
  epoch-0_step-10/                     # trains g21 + W21 with behavior anchor, negative 200-slice

outputs/v23_visual_transition/eval_comm21_step10_balanced_vllm_200_stop/
  eval_summary_*.json                  # 18.5 TSR / 66.87 StepSR on first 200

checkpoints/v23_comm_gate_ultraweak_anchor_from_best_short/
  epoch-0_step-10/                     # ultraweak comm gate, still negative 200-slice

outputs/v23_visual_transition/eval_comm_gate_ultraweak_step10_balanced_vllm_200_stop/
  eval_summary_*.json                  # 18.0 TSR / 67.20 StepSR on first 200

outputs/v23_visual_transition/full_sft_step250_hard_states/
  hard_states.jsonl                   # 1181 hard rows from full SFT step250 eval
  hard_state_summary.json

outputs/v23_visual_transition/full_sft_step250_hard_replay/
  hard_action_replay.jsonl            # weighted replay examples for first-error states
  hard_replay_summary.json
```

## Commands

Build offline action and transition examples:

```bash
python v23_visual_transition/prepare_offline_data.py \
  --input datasets/gui360-balanced/gui360_test_1000_balanced.jsonl \
  --output_dir outputs/v23_visual_transition/gui360_test_offline_data
```

Convert train parquet to episode JSONL and local PNG files:

```bash
.venv-qwen3-vllm/bin/python v23_visual_transition/convert_parquet_to_episode_jsonl.py \
  --parquet 'datasets/gui360-balanced/data/train-*.parquet' \
  --output_jsonl datasets/gui360-balanced/gui360_train_from_parquet.jsonl \
  --image_root datasets/gui360-balanced/images \
  --split_name train
```

Build train offline action and transition examples:

```bash
python v23_visual_transition/prepare_offline_data.py \
  --input datasets/gui360-balanced/gui360_train_from_parquet.jsonl \
  --output_dir outputs/v23_visual_transition/gui360_train_offline_data
```

Mine hard states from a completed eval:

```bash
python v23_visual_transition/mine_hard_states_from_eval.py \
  --eval_results outputs/gui360_fullparam_sft_step250_balanced_8gpu64_stop_bounded/eval_results_20260624_153846.json \
  --episode_data datasets/gui360-balanced/gui360_test_1000_balanced.jsonl \
  --output_dir outputs/v23_visual_transition/full_sft_step250_hard_states
```

Build weighted replay:

```bash
python v23_visual_transition/build_hard_replay.py \
  --hard_states outputs/v23_visual_transition/full_sft_step250_hard_states/hard_states.jsonl \
  --episode_data datasets/gui360-balanced/gui360_test_1000_balanced.jsonl \
  --output_dir outputs/v23_visual_transition/full_sft_step250_hard_replay
```

Sample matcher-scored candidates once a vLLM/OpenAI-compatible server is running:

```bash
python v23_visual_transition/sample_matcher_candidates.py \
  --hard_states outputs/v23_visual_transition/train_full_sft_hard_states/hard_states.jsonl \
  --episode_data datasets/gui360-balanced/gui360_train_from_parquet.jsonl \
  --output_dir outputs/v23_visual_transition/train_full_sft_k8_candidates \
  --api_url http://127.0.0.1:8000/v1 \
  --model_name gui360_fullparam_sft_step250 \
  --num_samples 8 \
  --threads 16
```

Train offline candidate GRPO from the fixed candidate set:

```bash
mkdir -p outputs/v23_visual_transition/offline_candidate_grpo_full_sft

OUT_DIR=checkpoints/v23_offline_candidate_grpo_full_sft \
LOG_DIR=outputs/v23_visual_transition/offline_candidate_grpo_full_sft \
CANDIDATE_DATA=outputs/v23_visual_transition/train_full_sft_k8_candidates/matcher_candidates.jsonl \
EPISODE_DATA=datasets/gui360-balanced/gui360_train_from_parquet.jsonl \
MODEL_PATH=checkpoints/gui360-fullparam-sft-step250 \
NPROC=8 \
MASTER_PORT=29561 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nohup bash v23_visual_transition/scripts/train_offline_candidate_grpo.sh \
  > outputs/v23_visual_transition/offline_candidate_grpo_full_sft/run.log 2>&1 &
```

Build WHAT/WHERE expert decomposition data:

```bash
python v23_visual_transition/build_where_what_dataset.py \
  --episode_data datasets/gui360-balanced/gui360_train_from_parquet.jsonl \
  --candidate_data outputs/v23_visual_transition/train_full_sft_k8_candidates/matcher_candidates.jsonl \
  --output_dir outputs/v23_visual_transition/where_what_train_dataset \
  --candidate_reward_threshold 0.5 \
  --max_success_candidates_per_state 1
```

## First Training Target

Use fixed matcher-scored candidates for the main offline RL update. Each hard state is a group: sampled candidate actions get matcher rewards, the GT action is inserted as a conservative anchor, and the trainer applies group-relative policy loss without calling `generate()` during training.

`hard_action_replay.jsonl` plus `offline_action_examples.jsonl` remains the SFT fallback/anchor data. Add `offline_transition_examples.jsonl` as a small auxiliary mix if the training stack supports two-image examples.

The current better direction is WHAT/WHERE specialization rather than actor RL. Use `what_examples.jsonl` to train the action-semantics expert and `where_examples.jsonl` to train the visual-grounding expert. `paired_where_what_examples.jsonl` preserves the joined state, full tool call, and split targets for a custom trainer that routes losses to the two cooperative LoRA slots.

Routed SFT trains the full `<tool_call>` target while supervising router tokens: function/status/non-location tokens route toward expert 1, and coordinate/start/end spans route toward expert 2. The smoke command:

```bash
PYTHONPATH="$PWD:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29575 \
.venv-qwen3-vllm/bin/torchrun --nproc_per_node=1 \
  v23_visual_transition/train_where_what_routed_sft.py \
  --model_path checkpoints/gui360-fullparam-sft-step250 \
  --train_data outputs/v23_visual_transition/where_what_train_dataset/paired_where_what_examples.jsonl \
  --output_dir checkpoints/v23_where_what_routed_sft_smoke \
  --max_rows 8 --max_steps 2 --gradient_accumulation_steps 1 \
  --save_steps 1 --logging_steps 1 --num_workers 0 \
  --route_loss_weight 0.2 --balance_weight 0.0 --weight_clip 2.0
```

Current best result: `checkpoints/v23_where_what_route_only_from_step5_short_fullsave/epoch-0_step-10` evaluated on the full balanced 1000 split reaches 22.3% TSR / 69.07% StepSR / 35.21% progress, above both the routed step-5 checkpoint (22.1% TSR / 68.49% StepSR / 35.02% progress) and the full SFT baseline (21.7% TSR / 68.22% StepSR / 34.77% progress). This checkpoint starts from routed step-5 and trains only the route weights for 10 more steps, with full cooperative checkpoint saving enabled.

Negative follow-up: `where_heavy` from the route-only step-10 checkpoint (`route + lora_A_2`, route loss 0.1, 10 steps) underperforms on the first 200 balanced episodes: 18.0% TSR / 66.67% StepSR / 31.73% progress versus route-only step-10 at 18.5% TSR / 67.72% StepSR / 32.48% progress. Do not spend full 1000 eval on this branch unless a later slice recovers.

Negative follow-up: continuing route-only from the same step-10 checkpoint with lower route loss 0.05 for 10 more steps also underperforms on the first 200 balanced episodes: 18.0% TSR / 66.94% StepSR / 31.71% progress. This suggests the useful route-only calibration window ended around the first 10 continuation steps from routed step-5.

Neutral/negative follow-up: `train_route_pairwise.py` implements frozen-expert route-only pairwise preference training from routed step-5 using GT positives and valid matcher-failing hard negatives (`click->click`, `type->click`, `click->type`). The first 10-step run did not beat route-only step-10 on the first 200 balanced episodes: step-5 checkpoint gets 18.0% TSR / 67.20% StepSR / 31.92% progress, and step-10 gets 18.5% TSR / 66.73% StepSR / 31.65% progress, below route-only step-10's 18.5% TSR / 67.72% StepSR / 32.48% progress. The script remains useful for future pair-selection/objective variants, but this exact objective should not be full-evaluated.

Comm-gate diagnostic: current-best communication gates carry almost no WHAT/WHERE role information. On 128 train rows, all global gate means sit near 0.501 and WHERE `(g21-g12)` is only 0.00004. `comm_gate_only` training from current best can mechanically open the intended WHAT-to-WHERE channel: step-10 raises WHERE `(g21-g12)` to 0.0203. However, this direct directional objective underperforms route-only step-10 on the first 200 balanced episodes: step-5 gets 18.0% TSR / 66.53% StepSR / 31.34% progress, and step-10 gets 18.5% TSR / 67.00% StepSR / 32.06% progress. Do not full-eval this exact comm-gate objective.

Behavior-anchored comm-gate follow-up: `comm_gate_only` with LM loss disabled and a token-logprob anchor to the initial route-only step-10 checkpoint improves over direct comm-gate step-10 but still remains below current best. First 200 balanced episodes: anchored step-5 gets 17.5% TSR / 66.19% StepSR / 31.16% progress; anchored step-10 gets 18.5% TSR / 67.20% StepSR / 32.25% progress. Current best route-only step-10 remains 18.5% TSR / 67.72% StepSR / 32.48% progress on the same slice. Do not full-eval this branch.

Comm message follow-up: `comm_21_only` lets WHAT-to-WHERE gate and message matrix (`comm_gate_21`, `comm_W_21`) move together under the same token-logprob behavior anchor. This did not help: step-10 gets 18.5% TSR / 66.87% StepSR / 31.96% progress on the first 200 balanced episodes, worse than anchored gate-only and current best. Do not full-eval this branch.

Ultraweak comm-gate follow-up: reducing the directional pressure (`margin=0.005`, `comm_loss_weight=0.2`, stronger behavior anchor) keeps gate movement much smaller but still does not recover the current best: step-10 gets 18.0% TSR / 67.20% StepSR / 31.95% progress on the first 200 balanced episodes. This suggests post-hoc communication-gate shaping is not the right improvement lever unless the communication objective is tied to actual outcome-correcting states rather than global WHERE-token role labels.

Primary diagnostic: first-error depth, premature terminate rate, progress, TSR on 200-episode slices before full 1000 eval.

Current conclusion for communication experiments: the current best has near-constant comm gates, and supervised gate shaping can open the WHAT-to-WHERE channel mechanically, but every tested comm-gate variant underperforms route-only step-10 on the 200-slice. Do not continue global comm-gate role supervision. If revisiting communication, use targeted hard-state supervision tied to fixed first-error repairs, not all coordinate tokens.

Done: train hard states, hard replay, K8 matcher candidates, WHAT/WHERE routed SFT, route-only continuation, route-pairwise probing, comm-gate information probing, and behavior-anchored comm-gate probing are available. The online/offline GRPO path is not the current best direction because candidate actor updates damaged output format. Next experiments should avoid continuing from route-only step-10 directly and should not full-eval the current pairwise or comm-gate-only runs. The next promising direction is no longer simply opening comm gates globally; it should either target comm-gate updates only on states current-best gets wrong, or use a much stronger teacher constraint / selective layer schedule, with 200-episode slices before any full 1000 eval.