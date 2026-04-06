#!/bin/bash
# ============================================
# Submit all SFT v3 exp2 jobs
# Phase 1: Step 1 + Step 3 + Step 4 (parallel)
# Phase 2: Step 2 classification (depends on Step 1)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/sft_v3"

# Create result directories
mkdir -p $RESULTS_DIR/{ac,gui360,classification,analysis}
mkdir -p $SCRIPT_DIR/logs

echo "============================================"
echo "Submitting SFT v3 Exp2 Jobs"
echo "Model: gui360_lora_sft_v3_merged"
echo "============================================"

# Phase 1: Independent evaluation jobs
echo ""
echo "--- Phase 1: Evaluation Jobs ---"

JOB_S1_G360=$(sbatch --parsable $SCRIPT_DIR/sft_v3_step1_gui360_nostop.slurm)
echo "Step 1 GUI-360 nostop: Job $JOB_S1_G360"

JOB_S1_AC=$(sbatch --parsable $SCRIPT_DIR/sft_v3_step1_ac_nostop.slurm)
echo "Step 1 AC nostop:     Job $JOB_S1_AC"

JOB_S3_G360=$(sbatch --parsable $SCRIPT_DIR/sft_v3_step3_gui360_summary.slurm)
echo "Step 3 GUI-360 summary: Job $JOB_S3_G360"

JOB_S3_AC=$(sbatch --parsable $SCRIPT_DIR/sft_v3_step3_ac_summary.slurm)
echo "Step 3 AC summary:     Job $JOB_S3_AC"

JOB_S4_G360=$(sbatch --parsable $SCRIPT_DIR/sft_v3_step4_gui360_subtask.slurm)
echo "Step 4 GUI-360 subtask: Job $JOB_S4_G360"

JOB_S4_AC=$(sbatch --parsable $SCRIPT_DIR/sft_v3_step4_ac_subtask.slurm)
echo "Step 4 AC subtask:     Job $JOB_S4_AC"

# Phase 2: Classification (depends on Step 1)
echo ""
echo "--- Phase 2: Classification (after Step 1) ---"

# Update the dependency in the classify script dynamically
CLASSIFY_SCRIPT="$SCRIPT_DIR/sft_v3_step2_classify.slurm"
JOB_S2=$(sbatch --parsable --dependency=afterok:${JOB_S1_G360}:${JOB_S1_AC} $CLASSIFY_SCRIPT)
echo "Step 2 Classification: Job $JOB_S2 (after $JOB_S1_G360,$JOB_S1_AC)"

echo ""
echo "============================================"
echo "All SFT v3 jobs submitted!"
echo "============================================"
echo ""
echo "Phase 1 (parallel): $JOB_S1_G360, $JOB_S1_AC, $JOB_S3_G360, $JOB_S3_AC, $JOB_S4_G360, $JOB_S4_AC"
echo "Phase 2 (depends):  $JOB_S2"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results in:   $RESULTS_DIR"
