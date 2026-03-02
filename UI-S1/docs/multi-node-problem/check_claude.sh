#!/bin/bash
# Quick script to monitor Claude's background progress

cd /lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

echo "=========================================="
echo "Claude Background Monitor"
echo "Time: $(date)"
echo "=========================================="

echo -e "\n=== Claude Process ==="
ps aux | grep -E "claude.*dangerous" | grep -v grep || echo "Claude not running"

echo -e "\n=== SLURM Jobs ==="
squeue -u shuqing.a5l 2>/dev/null || echo "No jobs"

echo -e "\n=== Last 30 lines of Claude output ==="
tail -30 claude_debug.log 2>/dev/null || echo "No log yet"

echo -e "\n=== Change Log (last 50 lines) ==="
tail -50 docs/change_log.md 2>/dev/null || echo "No change log yet"

echo -e "\n=== Latest test job logs ==="
latest_log=$(ls -t train/logs/test_offload_*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "File: $latest_log"
    echo "--- Last 20 lines ---"
    tail -20 "$latest_log"
else
    echo "No test logs found"
fi
