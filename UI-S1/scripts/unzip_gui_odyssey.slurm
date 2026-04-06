#!/bin/bash
#SBATCH --job-name=unzip_gui_odyssey
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/unzip_gui_odyssey_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/unzip_gui_odyssey_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

set -ex

TARGET_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-Odyssey"
EXTRACT_DIR="$TARGET_DIR/data/screenshots"

mkdir -p $EXTRACT_DIR

echo "Start unzip: $(date)"
unzip $TARGET_DIR/screenshots/merged_screenshots.zip -d $EXTRACT_DIR

echo "Cleaning up merged zip..."
rm $TARGET_DIR/screenshots/merged_screenshots.zip

echo "=========================================="
echo "Done at $(date)"
echo "Screenshot count: $(find $EXTRACT_DIR -name '*.png' | wc -l)"
du -sh $TARGET_DIR/
df -h /scratch/a5l/shuqing.a5l/
echo "=========================================="
