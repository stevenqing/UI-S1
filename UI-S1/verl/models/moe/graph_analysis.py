# Copyright 2024 UI-S1 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Spectral graph analysis for option discovery on GUI-360.

Implements the neural eigenfunction approximation from:
- Wu et al. (2019), "The Laplacian in RL: Learning Representations with
  Efficient Approximations", ICLR 2019.
- Jinnai et al. (2020), "Exploration in RL with Deep Covering Options", ICLR 2020.

The core idea: train a neural network f_net to approximate the second
eigenvector of the graph Laplacian. States with extreme f values (low percentile)
are connectivity bottlenecks — the hardest-to-reach UI states.

Task 3 in the Option-Incentivized MoE pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EigenfunctionConfig:
    """Configuration for eigenfunction training."""
    # Network architecture
    input_dim: int = 43         # state embedding dimensionality
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.0

    # Training
    num_epochs: int = 200
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-5
    eta: float = 1.0            # Lagrange multiplier for repulsive term

    # Bottleneck identification
    percentile_k: float = 30.0  # Desktop GUI: k=30 (Instruction §5.4)

    # Per-app training (recommended for GUI-360)
    per_app: bool = True        # Train separate f_net per app domain
    app_filter: Optional[str] = None  # If set, only train on this app

    # Checkpointing
    save_every: int = 50        # Save checkpoint every N epochs
    log_every: int = 10         # Log metrics every N epochs

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "eta": self.eta,
            "percentile_k": self.percentile_k,
            "per_app": self.per_app,
            "app_filter": self.app_filter,
        }


# ============================================================================
# Neural Eigenfunction Network
# ============================================================================

class EigenfunctionNet(nn.Module):
    """MLP that approximates the second eigenvector of the graph Laplacian.

    Maps a state embedding (R^d) to a scalar f-value (R^1).
    States with extreme f-values are connectivity bottlenecks.
    """

    def __init__(
        self,
        input_dim: int = 43,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map state embeddings to scalar f-values.

        Args:
            x: (batch_size, input_dim) state embeddings.

        Returns:
            (batch_size, 1) f-values.
        """
        return self.net(x)


# ============================================================================
# Dataset
# ============================================================================

class TransitionPairDataset(Dataset):
    """Dataset of (src_embedding, dst_embedding) transition pairs.

    Loads from the transition_pairs.npz file produced by Task 2.
    """

    def __init__(
        self,
        src_embeddings: np.ndarray,
        dst_embeddings: np.ndarray,
    ):
        assert src_embeddings.shape == dst_embeddings.shape
        self.src = torch.from_numpy(src_embeddings).float()
        self.dst = torch.from_numpy(dst_embeddings).float()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.dst[idx]

    @classmethod
    def from_npz(cls, path: str | Path) -> "TransitionPairDataset":
        data = np.load(path)
        return cls(data["src_embeddings"], data["dst_embeddings"])


class StateEmbeddingDataset(Dataset):
    """Dataset of all unique state embeddings (for random sampling in repulsive term)."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings).float()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

    @classmethod
    def from_npz(cls, path: str | Path) -> "StateEmbeddingDataset":
        data = np.load(path, allow_pickle=True)
        return cls(data["embeddings"])


# ============================================================================
# Loss Function (Wu et al. 2019, Eq. 5)
# ============================================================================

def eigenfunction_loss(
    f_net: EigenfunctionNet,
    s: torch.Tensor,
    s_next: torch.Tensor,
    s_rand: torch.Tensor,
    eta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the unconstrained eigenfunction objective for the second eigenvector.

    G_tilde(f) = 0.5 * E[(f(s) - f(s'))^2]
                 + eta * [(E[f^2] - 1)^2 + (E[f])^2]

    Term 1 (smoothness): connected states should have similar f values.
    Term 2 (repulsive): enforces normalization (E[f^2]=1) and orthogonality
        to the constant first eigenvector (E[f]=0). Without the (E[f])^2 term,
        the loss has a spurious minimum at the constant f(s) = ±1/√2.

    At the true second eigenvector: smoothness = λ₂, repulsive = 0.
    At a constant solution: smoothness = 0, repulsive > 0 (penalized).

    Args:
        f_net: The eigenfunction network.
        s, s_next: Transition pairs (connected states).
        s_rand: Randomly sampled states (for repulsive term).
        eta: Weight of the repulsive term.

    Returns:
        (loss, metrics_dict) where metrics_dict contains component losses.
    """
    f_s = f_net(s)
    f_s_next = f_net(s_next)

    # Smoothness: connected states have similar f values
    smoothness = 0.5 * ((f_s - f_s_next) ** 2).mean()

    # Repulsive: enforce E[f^2] = 1 (normalization) and E[f] = 0 (orthogonality)
    f_r = f_net(s_rand)
    norm_penalty = (f_r.pow(2).mean() - 1).pow(2)   # (E[f²] - 1)²
    ortho_penalty = f_r.mean().pow(2)                 # (E[f])²
    repulsive = norm_penalty + ortho_penalty

    loss = smoothness + eta * repulsive

    metrics = {
        "loss": loss.item(),
        "smoothness": smoothness.item(),
        "repulsive": repulsive.item(),
        "norm_penalty": norm_penalty.item(),
        "ortho_penalty": ortho_penalty.item(),
        "f_mean": f_s.mean().item(),
        "f_std": f_s.std().item(),
        "f_min": f_s.min().item(),
        "f_max": f_s.max().item(),
    }

    return loss, metrics


# ============================================================================
# Training
# ============================================================================

def train_eigenfunction(
    transition_pairs_path: str | Path,
    state_embeddings_path: str | Path,
    config: EigenfunctionConfig,
    output_dir: str | Path,
    device: str = "auto",
    state_registry_path: Optional[str | Path] = None,
) -> tuple[EigenfunctionNet, dict]:
    """Train the eigenfunction network on transition data.

    Args:
        transition_pairs_path: Path to transition_pairs.npz (from Task 2).
        state_embeddings_path: Path to all_state_embeddings.npz (from Task 2).
        config: Training configuration.
        output_dir: Where to save checkpoints and results.
        device: "auto", "cuda", or "cpu".
        state_registry_path: Optional path to state_registry.json (for per-app filtering).

    Returns:
        (trained_f_net, results_dict)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # --- Load data ---
    pairs = np.load(transition_pairs_path)
    src_emb = pairs["src_embeddings"]
    dst_emb = pairs["dst_embeddings"]

    states_data = np.load(state_embeddings_path, allow_pickle=True)
    all_state_emb = states_data["embeddings"]
    all_state_hashes = states_data["hashes"]

    # --- Per-app filtering ---
    if config.app_filter and state_registry_path:
        src_emb, dst_emb, all_state_emb, all_state_hashes = _filter_by_app(
            src_emb, dst_emb, all_state_emb, all_state_hashes,
            config.app_filter, state_registry_path,
        )

    logger.info(
        f"Data: {len(src_emb)} transition pairs, "
        f"{len(all_state_emb)} unique states, "
        f"embedding dim = {src_emb.shape[1]}"
    )

    # --- Create datasets ---
    transition_dataset = TransitionPairDataset(src_emb, dst_emb)
    transition_loader = DataLoader(
        transition_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    all_states_tensor = torch.from_numpy(all_state_emb).float().to(device)

    # --- Create model ---
    f_net = EigenfunctionNet(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        f_net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    # --- Save config ---
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # --- Training loop ---
    logger.info(f"Starting training for {config.num_epochs} epochs")
    history = []
    t0 = time.time()

    for epoch in range(1, config.num_epochs + 1):
        f_net.train()
        epoch_metrics = _train_one_epoch(
            f_net, optimizer, transition_loader, all_states_tensor,
            config, device,
        )
        scheduler.step()
        epoch_metrics["epoch"] = epoch
        epoch_metrics["lr"] = scheduler.get_last_lr()[0]
        history.append(epoch_metrics)

        if epoch % config.log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:4d}/{config.num_epochs}  "
                f"loss={epoch_metrics['loss']:.4f}  "
                f"smooth={epoch_metrics['smoothness']:.4f}  "
                f"repul={epoch_metrics['repulsive']:.4f}  "
                f"f_std={epoch_metrics['f_std']:.4f}  "
                f"[{elapsed:.0f}s]"
            )

        if epoch % config.save_every == 0:
            _save_checkpoint(f_net, optimizer, epoch, output_dir / f"checkpoint_epoch{epoch}.pt")

    total_time = time.time() - t0
    logger.info(f"Training complete in {total_time:.1f}s ({total_time/60:.1f}min)")

    # --- Save final model ---
    _save_checkpoint(f_net, optimizer, config.num_epochs, output_dir / "f_net_final.pt")

    # --- Compute f-values for all states ---
    f_net.eval()
    with torch.no_grad():
        f_values = f_net(all_states_tensor).cpu().numpy().flatten()

    # --- Identify bottlenecks ---
    bottleneck_info = identify_bottlenecks(
        f_values, all_state_hashes, config.percentile_k
    )

    # --- Save results ---
    results = {
        "config": config.to_dict(),
        "training_time_seconds": round(total_time, 1),
        "num_transitions": len(src_emb),
        "num_states": len(all_state_emb),
        "device": device,
        "final_loss": history[-1]["loss"],
        "final_smoothness": history[-1]["smoothness"],
        "final_repulsive": history[-1]["repulsive"],
        "f_value_stats": {
            "mean": float(f_values.mean()),
            "std": float(f_values.std()),
            "min": float(f_values.min()),
            "max": float(f_values.max()),
        },
        "bottleneck_threshold": bottleneck_info["threshold"],
        "num_bottleneck_states": bottleneck_info["num_bottleneck_states"],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save f-values
    np.savez(
        output_dir / "f_values.npz",
        hashes=all_state_hashes,
        f_values=f_values,
    )

    # Save bottleneck info
    with open(output_dir / "bottlenecks.json", "w") as f:
        json.dump(bottleneck_info, f, indent=2)

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f)

    logger.info(
        f"Results saved to {output_dir}. "
        f"Bottlenecks: {bottleneck_info['num_bottleneck_states']} states "
        f"(threshold={bottleneck_info['threshold']:.4f})"
    )

    return f_net, results


def _train_one_epoch(
    f_net: EigenfunctionNet,
    optimizer: torch.optim.Optimizer,
    transition_loader: DataLoader,
    all_states_tensor: torch.Tensor,
    config: EigenfunctionConfig,
    device: str,
) -> dict[str, float]:
    """Run one training epoch and return averaged metrics."""
    total_metrics: dict[str, float] = {}
    n_batches = 0

    for s_batch, s_next_batch in transition_loader:
        s_batch = s_batch.to(device)
        s_next_batch = s_next_batch.to(device)

        # Sample random states for repulsive term
        n = s_batch.size(0)
        idx = torch.randint(0, len(all_states_tensor), (n,))
        s_rand = all_states_tensor[idx]

        optimizer.zero_grad()
        loss, metrics = eigenfunction_loss(
            f_net, s_batch, s_next_batch, s_rand, config.eta
        )
        loss.backward()
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    # Average
    return {k: v / n_batches for k, v in total_metrics.items()}


def _filter_by_app(
    src_emb: np.ndarray,
    dst_emb: np.ndarray,
    all_state_emb: np.ndarray,
    all_state_hashes: np.ndarray,
    app: str,
    registry_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter data to a single app domain using the state registry."""
    with open(registry_path) as f:
        registry = json.load(f)

    app_hashes = {h for h, data in registry.items() if data["app_domain"] == app}
    logger.info(f"Filtering to app={app}: {len(app_hashes)} states in registry")

    # Filter unique states
    state_mask = np.array([str(h) in app_hashes for h in all_state_hashes])
    filtered_state_emb = all_state_emb[state_mask]
    filtered_state_hashes = all_state_hashes[state_mask]

    # Filter transitions: keep only transitions where src embedding matches
    # an app state. Since embeddings carry app one-hot, we can use dim 0-2.
    # app_domain index: excel=0, word=1, ppt=2
    app_idx = {"excel": 0, "word": 1, "ppt": 2}[app]
    trans_mask = src_emb[:, app_idx] > 0.5  # one-hot check
    filtered_src = src_emb[trans_mask]
    filtered_dst = dst_emb[trans_mask]

    logger.info(
        f"After filtering: {len(filtered_src)} transitions, "
        f"{len(filtered_state_emb)} states"
    )
    return filtered_src, filtered_dst, filtered_state_emb, filtered_state_hashes


# ============================================================================
# Bottleneck Identification
# ============================================================================

def identify_bottlenecks(
    f_values: np.ndarray,
    state_hashes: np.ndarray,
    percentile_k: float = 30.0,
) -> dict:
    """Identify bottleneck states from f-values.

    Bottleneck states are those with f-values below the k-th percentile.
    These are the hardest-to-reach states in the transition graph.

    Args:
        f_values: Array of f-values for each state.
        state_hashes: Array of state hash strings.
        percentile_k: Percentile threshold (lower = stricter).

    Returns:
        Dict with bottleneck info.
    """
    threshold = float(np.percentile(f_values, percentile_k))

    bottleneck_mask = f_values < threshold
    bottleneck_hashes = state_hashes[bottleneck_mask].tolist()
    bottleneck_f_values = f_values[bottleneck_mask].tolist()

    # Sort by f-value (most extreme first)
    sorted_pairs = sorted(zip(bottleneck_hashes, bottleneck_f_values), key=lambda x: x[1])

    return {
        "threshold": threshold,
        "percentile_k": percentile_k,
        "num_bottleneck_states": len(bottleneck_hashes),
        "num_total_states": len(f_values),
        "bottleneck_states": [
            {"hash": h, "f_value": round(fv, 6)} for h, fv in sorted_pairs
        ],
        "f_value_distribution": {
            "mean": float(f_values.mean()),
            "std": float(f_values.std()),
            "min": float(f_values.min()),
            "max": float(f_values.max()),
            "p10": float(np.percentile(f_values, 10)),
            "p25": float(np.percentile(f_values, 25)),
            "p50": float(np.percentile(f_values, 50)),
            "p75": float(np.percentile(f_values, 75)),
            "p90": float(np.percentile(f_values, 90)),
        },
    }


def map_bottlenecks_to_descriptions(
    bottleneck_info: dict,
    state_registry_path: str | Path,
) -> list[dict]:
    """Map bottleneck state hashes to human-readable UI state descriptions.

    Used for Task 4 validation — checking if identified bottlenecks match
    known hard UI transitions.

    Args:
        bottleneck_info: Output from identify_bottlenecks().
        state_registry_path: Path to state_registry.json.

    Returns:
        List of dicts with hash, f_value, app_domain, dialog_state, tab_signature.
    """
    with open(state_registry_path) as f:
        registry = json.load(f)

    described = []
    for entry in bottleneck_info["bottleneck_states"]:
        h = entry["hash"]
        state_info = registry.get(h, {})
        described.append({
            "hash": h,
            "f_value": entry["f_value"],
            "app_domain": state_info.get("app_domain", "unknown"),
            "dialog_state": state_info.get("dialog_state", "unknown"),
            "active_tab_signature": state_info.get("active_tab_signature", ""),
            "control_fingerprint": state_info.get("control_fingerprint", ""),
        })

    return described


# ============================================================================
# Checkpoint I/O
# ============================================================================

def _save_checkpoint(
    f_net: EigenfunctionNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": f_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_f_net(
    checkpoint_path: str | Path,
    config: Optional[EigenfunctionConfig] = None,
    device: str = "cpu",
) -> EigenfunctionNet:
    """Load a trained f_net from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        config: Config to reconstruct the network. If None, uses defaults.
        device: Device to load onto.

    Returns:
        Loaded EigenfunctionNet in eval mode.
    """
    if config is None:
        config = EigenfunctionConfig()

    f_net = EigenfunctionNet(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    f_net.load_state_dict(checkpoint["model_state_dict"])
    f_net.eval()
    f_net.to(device)
    return f_net
