"""Prepare mixed AP + Grounding dataset for LoRA v4 training."""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def main():
    # Load training datasets
    print("Loading AP training data...")
    with open(DATA_DIR / "gui360_train.json") as f:
        ap_train = json.load(f)
    print(f"  AP train samples: {len(ap_train)}")

    print("Loading Grounding training data...")
    with open(DATA_DIR / "gui360_grounding_train.json") as f:
        grounding_train = json.load(f)
    print(f"  Grounding train samples: {len(grounding_train)}")

    # Mix and shuffle training data
    mixed_train = ap_train + grounding_train
    random.seed(42)
    random.shuffle(mixed_train)
    print(f"  Mixed train samples: {len(mixed_train)}")

    # Save mixed training data
    out_train = DATA_DIR / "gui360_mixed_train.json"
    with open(out_train, "w") as f:
        json.dump(mixed_train, f, ensure_ascii=False)
    print(f"  Saved to {out_train}")

    # Load validation datasets
    print("\nLoading AP validation data...")
    with open(DATA_DIR / "gui360_val.json") as f:
        ap_val = json.load(f)
    print(f"  AP val samples: {len(ap_val)}")

    print("Loading Grounding validation data...")
    with open(DATA_DIR / "gui360_grounding_val.json") as f:
        grounding_val = json.load(f)
    print(f"  Grounding val samples: {len(grounding_val)}")

    # Sample 100 from each for validation
    random.seed(42)
    ap_val_sample = random.sample(ap_val, min(100, len(ap_val)))
    grounding_val_sample = random.sample(grounding_val, min(100, len(grounding_val)))
    mixed_val = ap_val_sample + grounding_val_sample
    random.shuffle(mixed_val)
    print(f"  Mixed val samples: {len(mixed_val)} (100 AP + 100 Grounding)")

    # Save mixed validation data
    out_val = DATA_DIR / "gui360_mixed_val.json"
    with open(out_val, "w") as f:
        json.dump(mixed_val, f, ensure_ascii=False)
    print(f"  Saved to {out_val}")

    print("\nDone!")

if __name__ == "__main__":
    main()
