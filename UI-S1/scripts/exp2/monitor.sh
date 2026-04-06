#!/bin/bash
# ============================================
# Monitor Exp2 jobs
# Usage: bash monitor.sh [--loop]
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

# Load job IDs if available
if [ -f "$SCRIPT_DIR/job_ids.txt" ]; then
    source "$SCRIPT_DIR/job_ids.txt"
fi

print_status() {
    echo "=========================================="
    echo "Exp2 Status Report - $(date)"
    echo "=========================================="

    # SLURM queue status
    echo ""
    echo "=== SLURM Queue ==="
    squeue -u $USER --format="%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null | grep -E "exp2|JOBID" || echo "No exp2 jobs in queue"

    echo ""
    echo "=== Results Check ==="

    # Step 1: Error Cascade
    echo ""
    echo "--- Step 1: Error Cascade ---"
    if [ -f "$RESULTS_DIR/ac/ac_nostop_natural_cascade_summary.json" ]; then
        echo "  AC (natural_cascade): DONE"
        python3 -c "import json; d=json.load(open('$RESULTS_DIR/ac/ac_nostop_natural_cascade_summary.json')); print(f\"    Success Rate: {d['trajectory_success_rate']:.2f}%, Step-0 Fail: {d['step0_failure_rate']:.2f}%\")" 2>/dev/null || true
    else
        echo "  AC (natural_cascade): PENDING"
    fi

    if [ -f "$RESULTS_DIR/ac/ac_nostop_oracle_rescue_summary.json" ]; then
        echo "  AC (oracle_rescue): DONE"
        python3 -c "import json; d=json.load(open('$RESULTS_DIR/ac/ac_nostop_oracle_rescue_summary.json')); print(f\"    Success Rate: {d['trajectory_success_rate']:.2f}%, Step-0 Fail: {d['step0_failure_rate']:.2f}%\")" 2>/dev/null || true
    else
        echo "  AC (oracle_rescue): PENDING"
    fi

    # Check GUI-360 results (look for latest nostop directory)
    if ls $RESULTS_DIR/gui360/nostop_*/*summary*.json 1>/dev/null 2>&1; then
        echo "  GUI-360 (no-stop AR): DONE"
        for f in $RESULTS_DIR/gui360/nostop_*/*summary*.json; do
            echo "    $(basename $f)"
        done
    else
        echo "  GUI-360 (no-stop AR): PENDING"
    fi

    # Step 2: Error Classification
    echo ""
    echo "--- Step 2: Error Classification ---"
    if [ -f "$RESULTS_DIR/classification/classification_summary.json" ]; then
        echo "  Classification: DONE"
        python3 -c "import json; d=json.load(open('$RESULTS_DIR/classification/classification_summary.json')); [print(f'    {k}: {v.get(\"total_errors\",0)} errors classified') for k,v in d.items()]" 2>/dev/null || true
    else
        echo "  Classification: PENDING"
    fi

    if [ -f "$RESULTS_DIR/analysis/cross_dataset_cascade_report.md" ]; then
        echo "  Cascade Analysis: DONE"
    else
        echo "  Cascade Analysis: PENDING"
    fi

    # Step 3: Summary Context
    echo ""
    echo "--- Step 3: Summary Context ---"
    for fmt in action_level semantic_level progress_level; do
        if [ -f "$RESULTS_DIR/ac/ac_summary_${fmt}_summary.json" ]; then
            echo "  AC ($fmt): DONE"
            python3 -c "import json; d=json.load(open('$RESULTS_DIR/ac/ac_summary_${fmt}_summary.json')); print(f\"    Success Rate: {d['success_rate']:.2f}%\")" 2>/dev/null || true
        else
            echo "  AC ($fmt): PENDING"
        fi
    done

    if ls $RESULTS_DIR/gui360/summary_*/*summary*.json 1>/dev/null 2>&1; then
        echo "  GUI-360 (summary): DONE"
    else
        echo "  GUI-360 (summary): PENDING"
    fi

    # Step 4: Subtask Decomposition
    echo ""
    echo "--- Step 4: Subtask Decomposition ---"
    if [ -f "$RESULTS_DIR/ac/ac_subtask_eval_summary.json" ]; then
        echo "  AC (subtask): DONE"
        python3 -c "import json; d=json.load(open('$RESULTS_DIR/ac/ac_subtask_eval_summary.json')); print(f\"    Traj Success: {d['trajectory_success_rate']:.2f}%, Step Acc: {d['avg_step_accuracy']:.4f}\")" 2>/dev/null || true
    else
        echo "  AC (subtask): PENDING"
    fi

    if ls $RESULTS_DIR/gui360/subtask_*/*summary*.json 1>/dev/null 2>&1; then
        echo "  GUI-360 (subtask): DONE"
    else
        echo "  GUI-360 (subtask): PENDING"
    fi

    # Check for errors in recent logs
    echo ""
    echo "=== Recent Errors (last 5 lines of .err files) ==="
    for errfile in $LOG_DIR/*.err; do
        if [ -f "$errfile" ] && [ -s "$errfile" ]; then
            echo "  $(basename $errfile):"
            tail -5 "$errfile" 2>/dev/null | sed 's/^/    /'
        fi
    done

    echo ""
    echo "=========================================="
}

# Run once or loop
if [ "$1" == "--loop" ]; then
    while true; do
        clear
        print_status

        # Check if all done
        PENDING_JOBS=$(squeue -u $USER --format="%.20j" 2>/dev/null | grep "exp2" | wc -l)
        if [ "$PENDING_JOBS" -eq 0 ]; then
            echo ""
            echo "ALL JOBS COMPLETED!"
            echo ""
            # Print final summary
            echo "=== FINAL RESULTS ==="
            for f in $RESULTS_DIR/ac/*_summary.json $RESULTS_DIR/gui360/*_summary*.json $RESULTS_DIR/classification/*_summary.json $RESULTS_DIR/analysis/*.md; do
                if [ -f "$f" ]; then
                    echo ""
                    echo "--- $(basename $f) ---"
                    cat "$f" 2>/dev/null | head -30
                fi
            done
            break
        fi

        echo ""
        echo "Refreshing in 60 seconds... (Ctrl+C to stop)"
        sleep 60
    done
else
    print_status
fi
