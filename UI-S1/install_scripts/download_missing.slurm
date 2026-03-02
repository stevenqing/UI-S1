#!/bin/bash
#SBATCH --job-name=download_missing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=workq

# Download remaining missing images for UI-S1

set -e

eval "$(conda shell.bash hook)"
conda activate ui-s1

cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

echo "=== Downloading Missing Episodes ==="
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

# Get current missing episodes
downloaded = set(os.listdir(images_dir))
missing_episodes = set()

with open('/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/ui_s1_train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for step in d['steps']:
            img = os.path.basename(step['screenshot'])
            if img not in downloaded:
                ep_id = int(img.split('_')[0])
                missing_episodes.add(ep_id)

print(f"Looking for {len(missing_episodes)} missing episodes")
print(f"Episode IDs: {sorted(missing_episodes)}")

if not missing_episodes:
    print("No missing episodes!")
    exit(0)

min_ep = min(missing_episodes)
print(f"\nMinimum missing episode ID: {min_ep}")

def download_from_split(split_name, missing_eps):
    """Download missing episodes from a dataset split."""
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

            # Check if all found
            if found_episodes == missing_eps:
                print(f"\nAll missing episodes found in {split_name}!")
                break

    return found_episodes, saved_count

# Try train split first (higher episode IDs might be later in the dataset)
found_train, saved_train = download_from_split("train", missing_episodes)
print(f"Found {len(found_train)} episodes, saved {saved_train} images from train")

# Check what's still missing
still_missing = missing_episodes - found_train
if still_missing:
    print(f"\nStill missing {len(still_missing)} episodes: {sorted(still_missing)}")
    found_test, saved_test = download_from_split("test", still_missing)
    print(f"Found {len(found_test)} episodes, saved {saved_test} images from test")
    still_missing = still_missing - found_test

# Final report
print("\n=== Final Status ===")
downloaded_now = set(os.listdir(images_dir))
final_missing = set()
with open('/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/ui_s1_train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for step in d['steps']:
            img = os.path.basename(step['screenshot'])
            if img not in downloaded_now:
                final_missing.add(img)

print(f"Total images now: {len(downloaded_now)}")
print(f"Still missing images: {len(final_missing)}")

if final_missing:
    print(f"Sample missing: {list(final_missing)[:10]}")
else:
    print("All images downloaded successfully!")
PY

echo "=== Download complete ==="
date

# Regenerate filtered dataset
python - <<'PY'
import json
import os

downloaded = set(os.listdir('/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/AndroidControl/images'))

complete = 0
incomplete = 0

with open('/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/ui_s1_train.jsonl') as f:
    with open('/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/ui_s1_train_filtered.jsonl', 'w') as out:
        for line in f:
            d = json.loads(line)
            sample_complete = all(
                os.path.basename(step['screenshot']) in downloaded
                for step in d['steps']
            )
            if sample_complete:
                out.write(line)
                complete += 1
            else:
                incomplete += 1

print(f"Updated filtered dataset: {complete} complete, {incomplete} incomplete")
PY
