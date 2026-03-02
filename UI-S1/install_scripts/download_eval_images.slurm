#!/bin/bash
#SBATCH --job-name=download_eval_imgs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=workq

# =============================================================================
# Download Evaluation Images for UI-S1
# =============================================================================

set -e

eval "$(conda shell.bash hook)"
conda activate ui-s1

cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

echo "=== Downloading Evaluation Images ==="
date

python - <<'PY'
import os
import json
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

images_dir = '/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/AndroidControl/images'
eval_dataset = '/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/android_control_evaluation_std.jsonl'

os.makedirs(images_dir, exist_ok=True)

# Get required images for evaluation
required_images = set()
with open(eval_dataset) as f:
    for line in f:
        d = json.loads(line)
        for step in d['steps']:
            required_images.add(os.path.basename(step['screenshot']))

print(f"Total images needed for evaluation: {len(required_images)}")

# Check what's missing
downloaded = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
missing = required_images - downloaded

if not missing:
    print(f"All {len(required_images)} evaluation images already available!")
    exit(0)

print(f"Already have: {len(downloaded)}")
print(f"Missing: {len(missing)}")

# Get missing episode IDs
missing_episodes = set()
for img in missing:
    ep_id = int(img.split('_')[0])
    missing_episodes.add(ep_id)

print(f"Need to download from {len(missing_episodes)} episodes")
print(f"Episode ID range: {min(missing_episodes)} - {max(missing_episodes)}")

def download_from_split(split_name, missing_eps):
    print(f"\n=== Checking {split_name} split ===")
    dataset = load_dataset("smolagents/android-control", split=split_name, streaming=True)

    saved_count = 0
    found_episodes = set()

    for example in tqdm(dataset, desc=f"Processing {split_name}"):
        episode_id = example['episode_id']

        if episode_id in missing_eps:
            found_episodes.add(episode_id)
            screenshots = example['screenshots_b64']

            for i, screenshot_b64 in enumerate(screenshots):
                img_filename = f"{episode_id}_{i}.png"
                img_path = os.path.join(images_dir, img_filename)

                if os.path.exists(img_path):
                    continue

                try:
                    img_data = base64.b64decode(screenshot_b64)
                    img = Image.open(BytesIO(img_data))
                    img.save(img_path)
                    saved_count += 1
                except Exception as e:
                    print(f"Error saving {img_filename}: {e}")

            if found_episodes == missing_eps:
                print(f"\nAll missing episodes found in {split_name}!")
                break

    return found_episodes, saved_count

# Download from train split first
found_train, saved_train = download_from_split("train", missing_episodes)
print(f"Found {len(found_train)} episodes, saved {saved_train} images from train")

# Check what's still missing
still_missing = missing_episodes - found_train
if still_missing:
    print(f"\nStill missing {len(still_missing)} episodes, checking test split...")
    found_test, saved_test = download_from_split("test", still_missing)
    print(f"Found {len(found_test)} episodes, saved {saved_test} images from test")
    still_missing = still_missing - found_test

# Final verification
print("\n=== Final Verification ===")
downloaded_now = set(os.listdir(images_dir))
still_missing_imgs = required_images - downloaded_now

print(f"Total images now: {len(downloaded_now)}")
print(f"Required for eval: {len(required_images)}")
print(f"Still missing: {len(still_missing_imgs)}")

if still_missing_imgs:
    missing_eps = set(int(img.split('_')[0]) for img in still_missing_imgs)
    print(f"Missing episode IDs: {sorted(missing_eps)[:20]}...")
else:
    print("All evaluation images downloaded successfully!")
PY

echo "=== Download complete ==="
date
