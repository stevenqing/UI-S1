#!/bin/bash
#SBATCH --job-name=download_images
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=workq

# =============================================================================
# Download AndroidControl Images for UI-S1 Training
# =============================================================================

set -e

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate ui-s1

DATASET_DIR=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets
IMAGES_DIR=$DATASET_DIR/AndroidControl/images
mkdir -p $IMAGES_DIR

cd $DATASET_DIR

echo "=== Downloading AndroidControl Images ==="
echo "This will download images from smolagents/android-control dataset"
echo "Target directory: $IMAGES_DIR"

python - <<'PY'
import os
import json
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Load the UI-S1 dataset to get required image paths
print("Loading UI-S1 dataset to find required images...")
required_images = set()
with open('/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/ui_s1_train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for step in d['steps']:
            # Extract episode ID from path like /images/0_0.png -> 0
            img_path = step['screenshot']
            required_images.add(img_path)

print(f"Required images: {len(required_images)}")

# Parse required episode IDs
required_episodes = set()
for img in required_images:
    # /images/0_0.png -> episode_id = 0
    basename = os.path.basename(img)
    episode_id = int(basename.split('_')[0])
    required_episodes.add(episode_id)

print(f"Required episodes: {len(required_episodes)}")
print(f"Sample episode IDs: {sorted(required_episodes)[:10]}")

# Create images directory
images_dir = '/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/AndroidControl/images'
os.makedirs(images_dir, exist_ok=True)

# Load android-control dataset
print("\nLoading android-control dataset from HuggingFace...")
print("This may take a while as the dataset is ~67GB...")

try:
    dataset = load_dataset("smolagents/android-control", split="train", streaming=True)

    saved_count = 0
    checked_count = 0

    for example in tqdm(dataset, desc="Processing episodes"):
        checked_count += 1
        episode_id = example['episode_id']

        # Check if we need this episode
        if episode_id in required_episodes:
            screenshots = example['screenshots_b64']

            for i, screenshot_b64 in enumerate(screenshots):
                img_filename = f"{episode_id}_{i}.png"
                img_path = os.path.join(images_dir, img_filename)

                # Skip if already exists
                if os.path.exists(img_path):
                    continue

                try:
                    # Decode base64 and save image
                    img_data = base64.b64decode(screenshot_b64)
                    img = Image.open(BytesIO(img_data))
                    img.save(img_path)
                    saved_count += 1
                except Exception as e:
                    print(f"Error saving {img_filename}: {e}")

            # Remove from required set
            required_episodes.discard(episode_id)

            if len(required_episodes) == 0:
                print(f"\nAll required episodes found! Checked {checked_count} episodes.")
                break

        if checked_count % 1000 == 0:
            print(f"Checked {checked_count} episodes, saved {saved_count} images, {len(required_episodes)} episodes remaining")

    print(f"\nTotal images saved: {saved_count}")
    print(f"Images directory: {images_dir}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("\nTrying alternative approach with direct download...")

    # Alternative: Download from Google Cloud Storage TFRecords
    print("Please download manually from: https://github.com/google-research/google-research/tree/master/android_control")

# Verify downloaded images
print("\n=== Verifying downloaded images ===")
downloaded = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
print(f"Downloaded images: {len(downloaded)}")

# Check coverage
missing = []
for img in required_images:
    basename = os.path.basename(img)
    if basename not in downloaded:
        missing.append(img)

if missing:
    print(f"Missing images: {len(missing)}")
    print(f"Sample missing: {missing[:5]}")
else:
    print("All required images are available!")
PY

echo "=== Download complete ==="
ls -lh $IMAGES_DIR | head -20
echo "..."
echo "Total files: $(ls $IMAGES_DIR | wc -l)"
date
