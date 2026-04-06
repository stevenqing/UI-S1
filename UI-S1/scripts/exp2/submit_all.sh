#!/bin/bash
# ============================================
# Submit all Exp2 jobs
# Phase 1: Independent evaluations (parallel)
# Phase 2: Analysis (depends on Phase 1)
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p $LOG_DIR

echo "=========================================="
echo "Exp2: Cross-Dataset Analysis Submission"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========== PHASE 1: Independent evaluations ==========
echo "=== Phase 1: Submitting independent evaluations ==="

# Step 1: Error Cascade
JOB_S1_G360=$(sbatch --parsable $SCRIPT_DIR/step1_gui360_nostop.slurm)
echo "Step 1 GUI-360 (no-stop AR):    Job $JOB_S1_G360"

JOB_S1_AC=$(sbatch --parsable $SCRIPT_DIR/step1_ac_nostop.slurm)
echo "Step 1 AC (no-stop AR):          Job $JOB_S1_AC"

# Step 3: Summary Context
JOB_S3_G360=$(sbatch --parsable $SCRIPT_DIR/step3_gui360_summary.slurm)
echo "Step 3 GUI-360 (summary):        Job $JOB_S3_G360"

JOB_S3_AC=$(sbatch --parsable $SCRIPT_DIR/step3_ac_summary.slurm)
echo "Step 3 AC (summary):             Job $JOB_S3_AC"

# Step 4: Subtask Decomposition
JOB_S4_G360=$(sbatch --parsable $SCRIPT_DIR/step4_gui360_subtask.slurm)
echo "Step 4 GUI-360 (subtask):        Job $JOB_S4_G360"

JOB_S4_AC=$(sbatch --parsable $SCRIPT_DIR/step4_ac_subtask.slurm)
echo "Step 4 AC (subtask):             Job $JOB_S4_AC"

echo ""
echo "=== Phase 2: Submitting analysis (depends on Step 1) ==="

# Step 2: Error Classification + Cascade Analysis
# Fix the dependency placeholder in the slurm file
sed -i "s/STEP1_AC_JOBID/$JOB_S1_AC/g; s/STEP1_G360_JOBID/$JOB_S1_G360/g" \
    $SCRIPT_DIR/step2_classify.slurm 2>/dev/null || true

JOB_S2=$(sbatch --parsable --dependency=afterany:${JOB_S1_AC}:${JOB_S1_G360} $SCRIPT_DIR/step2_classify.slurm)
echo "Step 2 Classification+Analysis:  Job $JOB_S2 (after $JOB_S1_AC, $JOB_S1_G360)"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Job Summary:"
echo "  Phase 1 (parallel):"
echo "    $JOB_S1_G360  step1_gui360_nostop"
echo "    $JOB_S1_AC    step1_ac_nostop"
echo "    $JOB_S3_G360  step3_gui360_summary"
echo "    $JOB_S3_AC    step3_ac_summary"
echo "    $JOB_S4_G360  step4_gui360_subtask"
echo "    $JOB_S4_AC    step4_ac_subtask"
echo "  Phase 2 (after Phase 1):"
echo "    $JOB_S2       step2_classify+analysis"
echo ""
echo "Monitor: bash $SCRIPT_DIR/monitor.sh"
echo "Quick check: squeue -u \$USER --name=exp2"
echo ""

# Save job IDs for monitoring
cat > $SCRIPT_DIR/job_ids.txt << EOF
# Exp2 Job IDs - $(date)
S1_G360=$JOB_S1_G360
S1_AC=$JOB_S1_AC
S3_G360=$JOB_S3_G360
S3_AC=$JOB_S3_AC
S4_G360=$JOB_S4_G360
S4_AC=$JOB_S4_AC
S2_CLASSIFY=$JOB_S2
EOF

echo "Job IDs saved to $SCRIPT_DIR/job_ids.txt"
