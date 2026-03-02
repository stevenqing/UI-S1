#!/usr/bin/env python3
"""
Fix image paths in gui360_test_sft_matched.parquet

Original path format: datasets/GUI-360/test/image/success/{episode_id}/...
Correct path format:  datasets/GUI-360/test/image/{app}/{category}/success/{episode_id}/...

Category mapping based on episode_id pattern:
- {app}_1_* -> in_app
- {app}_3_* -> online
- {app}_4_* -> search
"""

import pandas as pd
import json
import re
import os
from tqdm import tqdm

def get_category_from_episode(episode_id):
    """Extract category from episode_id based on the middle number."""
    # Pattern: {app}_{category_num}_{id}
    match = re.match(r'(\w+)_(\d+)_\d+', episode_id)
    if match:
        app = match.group(1)
        category_num = match.group(2)
        category_map = {
            '1': 'in_app',
            '3': 'online',
            '4': 'search'
        }
        return app, category_map.get(category_num, 'in_app')
    return None, None

def fix_image_path(image_path):
    """Fix a single image path."""
    # Pattern: datasets/GUI-360/test/image/success/{episode_id}/{image_file}
    match = re.match(r'(datasets/GUI-360/test/image)/success/([^/]+)/(.+)', image_path)
    if match:
        base_path = match.group(1)
        episode_id = match.group(2)
        image_file = match.group(3)

        app, category = get_category_from_episode(episode_id)
        if app and category:
            new_path = f"{base_path}/{app}/{category}/success/{episode_id}/{image_file}"
            return new_path

    # Return original if no match
    return image_path

def fix_messages(messages_str):
    """Fix all image paths in a messages string."""
    if isinstance(messages_str, str):
        messages = json.loads(messages_str)
    else:
        messages = messages_str

    for msg in messages:
        if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
            for content in msg['content']:
                if content.get('type') == 'image' and 'image' in content:
                    content['image'] = fix_image_path(content['image'])

    return json.dumps(messages, ensure_ascii=False)

def main():
    input_file = '/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/data/gui360_test_sft_matched.parquet'
    output_file = '/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/data/gui360_test_sft_fixed.parquet'

    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Total samples: {len(df)}")

    # Show example before fix
    sample_msg = df.iloc[0]['messages']
    if isinstance(sample_msg, str):
        sample_msg = json.loads(sample_msg)
    for msg in sample_msg:
        if msg.get('role') == 'user':
            for content in msg['content']:
                if content.get('type') == 'image':
                    print(f"Before: {content['image']}")
                    print(f"After:  {fix_image_path(content['image'])}")
                    break
            break

    # Fix all messages
    print("\nFixing image paths...")
    fixed_messages = []
    for i, msg in enumerate(tqdm(df['messages'], desc="Processing")):
        fixed_messages.append(fix_messages(msg))
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1}/{len(df)} samples")
    df['messages'] = fixed_messages

    # Save fixed parquet
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False)

    # Verify
    print("\nVerifying fix...")
    df_verify = pd.read_parquet(output_file)
    sample_msg = df_verify.iloc[0]['messages']
    if isinstance(sample_msg, str):
        sample_msg = json.loads(sample_msg)
    for msg in sample_msg:
        if msg.get('role') == 'user':
            for content in msg['content']:
                if content.get('type') == 'image':
                    image_path = content['image']
                    full_path = f"/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/{image_path}"
                    exists = os.path.exists(full_path)
                    print(f"Verified path: {image_path}")
                    print(f"File exists: {exists}")
                    break
            break

    print(f"\nDone! Fixed parquet saved to: {output_file}")

if __name__ == "__main__":
    main()
