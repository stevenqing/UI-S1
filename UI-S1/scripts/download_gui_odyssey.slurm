#!/bin/bash
#SBATCH --job-name=dl_gui_odyssey
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/dl_gui_odyssey_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/dl_gui_odyssey_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

set -ex

# Ensure git-lfs is in PATH BEFORE any git operations
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/habitat-llm/bin:/usr/bin:$PATH"
git lfs version

TARGET_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-Odyssey"

echo "=========================================="
echo "Start: $(date)"
df -h /scratch/a5l/shuqing.a5l/
echo "=========================================="

# Fresh clone with LFS enabled (smudge will pull LFS files during checkout)
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/
git lfs install
GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/hflqf88888/GUIOdyssey GUI-Odyssey

echo "=========================================="
echo "Clone completed at $(date)"
ls -lh $TARGET_DIR/screenshots/
df -h /scratch/a5l/shuqing.a5l/
echo "=========================================="

# Extract screenshots
cd $TARGET_DIR/screenshots
zip -s 0 screenshots.zip --out merged_screenshots.zip
unzip merged_screenshots.zip -d $TARGET_DIR/data/screenshots/
rm merged_screenshots.zip

echo "=========================================="
echo "Done at $(date)"
du -sh $TARGET_DIR/
find $TARGET_DIR/data/screenshots -name '*.png' | wc -l
echo "=========================================="
