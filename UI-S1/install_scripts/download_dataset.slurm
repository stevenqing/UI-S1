#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=workq

# =============================================================================
# Download UI-S1 Training Dataset from HuggingFace
# =============================================================================

set -e

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate ui-s1

cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets

echo "=== Downloading UI-S1 Dataset ==="

# Download using huggingface_hub
python - <<'PY'
from datasets import load_dataset
import json
import os

print("Loading dataset from HuggingFace...")
dataset = load_dataset("mPLUG/UI_S1_dataset", split="train")

print(f"Dataset loaded: {len(dataset)} examples")
print(f"Features: {dataset.features}")

# Save as JSONL format
output_file = "ui_s1_train.jsonl"
print(f"Saving to {output_file}...")

with open(output_file, 'w') as f:
    for i, example in enumerate(dataset):
        f.write(json.dumps(example, ensure_ascii=False) + '\n')
        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{len(dataset)} examples")

print(f"Dataset saved to {output_file}")
print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")

# Also save the dataset in HuggingFace format for easy loading
dataset.save_to_disk("ui_s1_dataset")
print("Dataset also saved to ui_s1_dataset/ directory in HuggingFace format")
PY

echo "=== Download complete ==="
ls -lh /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/
date
