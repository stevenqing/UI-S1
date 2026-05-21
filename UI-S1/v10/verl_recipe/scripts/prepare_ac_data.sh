#!/bin/bash
# Convert AndroidControl JSONL data to verl-compatible parquet format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Input paths
TRAIN_JSONL="${PROJECT_DIR}/datasets/cooperative_thought_ac/ac_train_thought.jsonl"
VAL_JSONL="${PROJECT_DIR}/datasets/cooperative_thought_ac/ac_val_thought.jsonl"

# Output paths
OUTPUT_DIR="${PROJECT_DIR}/data"
mkdir -p "$OUTPUT_DIR"

echo "Converting training data..."
python -m v10.verl_recipe.coop_dataset \
    --input "$TRAIN_JSONL" \
    --output "$OUTPUT_DIR/ac_train.parquet"

echo "Converting validation data..."
python -m v10.verl_recipe.coop_dataset \
    --input "$VAL_JSONL" \
    --output "$OUTPUT_DIR/ac_val.parquet"

echo "Done! Data saved to $OUTPUT_DIR/"
echo "  Train: $OUTPUT_DIR/ac_train.parquet"
echo "  Val:   $OUTPUT_DIR/ac_val.parquet"
