#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-v20/phase0/data/proposal_sft/per_agent_2000}"

"$PYTHON_BIN" v20/phase0/memory_controller/build_per_agent_proposal_sft_data.py \
  --test_data v20/phase0/data/gui360-balanced-jsonl/test.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --min_reward 0.5 \
  --max_train_per_agent 2000 \
  --max_dev_per_agent 300 \
  --format sharegpt \
  --run type_recovery=v20/phase0/results/v13_template_ui_s1_mc_type_recovery_candidate_v1_gt_history_full_candidates_sharded \
  --run click_recovery=v20/phase0/results/v13_template_ui_s1_mc_click_recovery_candidate_v1_gt_history_full_candidates_sharded \
  --run swipe_navigation=v20/phase0/results/v13_template_ui_s1_mc_swipe_navigation_candidate_v1_gt_history_full_candidates_sharded \
  --run minimal_next_step=v20/phase0/results/v13_template_ui_s1_mc_minimal_next_step_candidate_v1_gt_history_full_candidates_sharded \
  --run escape_finish=v20/phase0/results/v13_template_ui_s1_mc_ui_escape_finish_guard_v1_gt_history_full_candidates_sharded \
  --run spreadsheet_formula=v20/phase0/results/v13_template_ui_s1_mc_spreadsheet_formula_candidate_v1_gt_history_full_candidates_sharded
