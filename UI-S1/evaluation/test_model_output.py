#!/usr/bin/env python3
"""Quick test to see actual model output format"""
import json
import os

# Read one sample from the dataset
dataset_path = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/android_control_evaluation_fixed.jsonl"
with open(dataset_path, 'r') as f:
    sample = json.loads(f.readline())

print("=== Sample Task ===")
print(f"Goal: {sample['goal'][:100]}...")
print(f"First step image: {sample['steps'][0]['screenshot']}")

# Import the predict function
from os_genesis_utils import predict
from PIL import Image

# Load image
image_path = sample['steps'][0]['screenshot']
image = Image.open(image_path)
print(f"Image size: {image.size}")

# Make prediction
print("\n=== Calling model... ===")
response = predict(
    model_name="OS-Genesis-7B",
    instruction=sample['goal'],
    low_instruction='',
    history='',
    image=image
)

print("\n=== Model Response ===")
print(response)
print("\n=== Response Type ===")
print(type(response))

# Try parsing
print("\n=== Attempting to parse ===")
if "action:" in response.lower():
    action_start = response.lower().find("action:")
    print(f"Found 'action:' at position {action_start}")
    action_str = response[action_start + len("action:"):].strip()
    print(f"Action string: {action_str[:200]}...")
else:
    print("No 'action:' keyword found in response")
    # Check for other patterns
    if "<|box_start|>" in response:
        print("Found special token format <|box_start|>")
    if "thought:" in response.lower():
        print("Found 'thought:' keyword")
