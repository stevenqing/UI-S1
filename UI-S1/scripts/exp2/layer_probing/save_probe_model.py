"""Retrain L26 correctness probe on ALL clean samples and serialize as pickle.

Uses the same pipeline as probe_v2.py: StandardScaler → PCA(256) → LogisticRegression.
Trains on all 956 clean samples (no CV split) for deployment in verifier inference.
"""

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="Save L26 correctness probe as pickle")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="scripts/exp2/results/layer_probing/sft_v2_20260320_130909",
        help="Directory containing layer_26_last.npy and labels_v2.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output pickle path (default: <data_dir>/correctness_probe_L26.pkl)",
    )
    parser.add_argument("--layer", type=int, default=26, help="Layer index to use")
    parser.add_argument("--n_components", type=int, default=256, help="PCA components")
    args = parser.parse_args()

    # Load data
    labels_path = os.path.join(args.data_dir, "labels_v2.json")
    npy_path = os.path.join(args.data_dir, f"layer_{args.layer}_last.npy")

    print(f"Loading labels from {labels_path}")
    with open(labels_path, "r") as f:
        labels = json.load(f)

    print(f"Loading hidden states from {npy_path}")
    X = np.load(npy_path)  # [N, 3584]
    print(f"  Shape: {X.shape}")

    # Get clean correctness subset
    clean_idx = [i for i, l in enumerate(labels) if l.get("correctness_clean", False)]
    y = np.array([1 if labels[i]["step_correct"] else 0 for i in clean_idx])

    print(f"Clean samples: {len(clean_idx)} / {len(labels)}")
    print(f"Class balance: {y.sum()} correct ({y.mean():.3f}), {len(y) - y.sum()} wrong ({1 - y.mean():.3f})")

    X_clean = X[clean_idx]

    # Pipeline: StandardScaler → PCA(256) → LogisticRegression
    print(f"\nTraining pipeline: StandardScaler → PCA({args.n_components}) → LogisticRegression")

    scaler = StandardScaler().fit(X_clean)
    X_scaled = scaler.transform(X_clean)

    pca = PCA(n_components=args.n_components, random_state=42).fit(X_scaled)
    X_reduced = pca.transform(X_scaled)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42).fit(X_reduced, y)

    # Sanity check: train accuracy
    train_preds = clf.predict(X_reduced)
    train_acc = (train_preds == y).mean()
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Majority baseline: {max(y.mean(), 1 - y.mean()):.4f}")
    print(f"Delta: {train_acc - max(y.mean(), 1 - y.mean()):.4f}")

    # Save as pickle
    output_path = args.output_path or os.path.join(args.data_dir, "correctness_probe_L26.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    probe_data = {
        "scaler": scaler,
        "pca": pca,
        "clf": clf,
        "layer": args.layer,
        "n_components": args.n_components,
        "n_train_samples": len(clean_idx),
        "class_balance": {"correct": int(y.sum()), "wrong": int(len(y) - y.sum())},
        "train_accuracy": float(train_acc),
    }
    joblib.dump(probe_data, output_path)
    print(f"\nSaved probe to {output_path}")

    # Verify reload
    probe_loaded = joblib.load(output_path)
    X_test = probe_loaded["scaler"].transform(X_clean[:5])
    X_test = probe_loaded["pca"].transform(X_test)
    probs = probe_loaded["clf"].predict_proba(X_test)
    print(f"Verification - first 5 P(correct): {probs[:, 1].tolist()}")


if __name__ == "__main__":
    main()
