#!/bin/bash
#SBATCH --job-name=download_gui360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/download_gui360_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/download_gui360_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# Set HuggingFace token to avoid rate limiting
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null)}"

# Set variables
DATASET_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360"
REPO_ID="vyokky/GUI-360"

echo "=========================================="
echo "Starting GUI-360 Dataset Download"
echo "Target directory: $DATASET_DIR"
echo "Start time: $(date)"
echo "=========================================="

# Create directory if not exists
mkdir -p $DATASET_DIR

# Download dataset (will resume from existing files)
# Using --max-workers=4 to avoid HuggingFace rate limiting (5000 requests per 5 min)
echo "Downloading GUI-360 dataset..."
huggingface-cli download $REPO_ID \
    --repo-type dataset \
    --local-dir $DATASET_DIR \
    --resume-download \
    --max-workers 4

DOWNLOAD_EXIT_CODE=$?

echo "=========================================="
echo "Download completed at $(date)"
echo "Exit code: $DOWNLOAD_EXIT_CODE"
echo "=========================================="

# Show final statistics
echo "Final statistics:"
echo "Total files: $(find $DATASET_DIR -type f \( -name '*.jsonl' -o -name '*.png' \) | wc -l)"
echo "Total size: $(du -sh $DATASET_DIR | cut -f1)"

exit $DOWNLOAD_EXIT_CODE
