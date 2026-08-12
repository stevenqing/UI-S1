import hashlib
import json
import math
import random
from copy import deepcopy

import numpy as np
import torch

from trivus_assembly import with_model_weights
from trivus_data import (
    FAMILIES, INPUT_DIMENSION, fit_standardizer, stable_row_seed, torch_batch,
)
from trivus_model import TriVUSSetRanker, permute_batch, trivus_loss


MODEL_SPECS = {
    "JOINT3": ("JOINT3", None, FAMILIES),
    "TARGET_ONLY_MIND2WEB": ("TARGET_ONLY", "mind2web", ("mind2web",)),
    "TARGET_ONLY_SCREENSPOT_PRO": (
        "TARGET_ONLY", "screenspot_pro", ("screenspot_pro",),
    ),
    "TARGET_ONLY_ANDROIDCONTROL": (
        "TARGET_ONLY", "androidcontrol", ("androidcontrol",),
    ),
    "JOINT2_NO_ANDROID": (
        "JOINT2_NO_ANDROID", None, ("mind2web", "screenspot_pro"),
    ),
    "NO_VISUAL": ("NO_VISUAL", None, FAMILIES),
    "RANDOM_ID_PLACEBO": ("RANDOM_ID_PLACEBO", None, FAMILIES),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_spec(spec_id):
    if spec_id not in MODEL_SPECS:
        raise ValueError(f"TriVUS unknown model spec: {spec_id}")
    variant, target_family, families = MODEL_SPECS[spec_id]
    return {
        "spec_id": spec_id,
        "variant": variant,
        "target_family": target_family,
        "families": tuple(families),
    }


def make_model(input_dimension=INPUT_DIMENSION):
    return TriVUSSetRanker(
        input_dimension, width=64, heads=4, layers=2, dropout=0.1
    )


def active_indices(data):
    indices = np.flatnonzero(data.active & (data.weights > 0))
    if not len(indices):
        raise ValueError("TriVUS fit has no active positive-weight rows")
    return indices


def epoch_order(sample_keys, indices, seed, epoch):
    return np.asarray(sorted(
        (int(index) for index in indices),
        key=lambda index: (
            stable_row_seed(sample_keys[index], seed, epoch),
            sample_keys[index],
        ),
    ), dtype=np.int64)


def epoch_permutations(sample_keys, seed, epoch):
    values = []
    for sample_key in sample_keys:
        derived = stable_row_seed(sample_key, seed, epoch)
        values.append(np.random.default_rng(derived).permutation(12))
    return np.stack(values).astype(np.int64)


def prepare_spec_data(data, spec):
    weighted = with_model_weights(
        data, spec["variant"], spec["target_family"]
    )
    return weighted, fit_standardizer(
        weighted, spec["variant"], included_families=spec["families"]
    )


def train_epoch(model, data, optimizer, seed, epoch, batch_size, gradient_clip, device):
    model.train()
    indices = active_indices(data)
    order = epoch_order(data.sample_keys, indices, seed, epoch)
    permutations = epoch_permutations(data.sample_keys, seed, epoch)
    normalization = torch.as_tensor(
        float(data.weights[indices].sum()), dtype=torch.float32, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(order), batch_size):
        selected = order[start:start + batch_size]
        batch = torch_batch(data, selected, device)
        permutation = torch.as_tensor(permutations[selected], device=device)
        batch = permute_batch(batch, permutation)
        loss, _ = trivus_loss(model, batch, normalization=normalization)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()


def evaluate_loss(model, data, batch_size, device):
    model.eval()
    indices = active_indices(data)
    normalization = torch.as_tensor(
        float(data.weights[indices].sum()), dtype=torch.float32, device=device
    )
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = torch_batch(data, indices[start:start + batch_size], device)
            loss, _ = trivus_loss(model, batch, normalization=normalization)
            total += float(loss)
    return total


def train_with_checkpoint(train_data, checkpoint_data, spec_id, config, seed, device):
    if (
        set(train_data.context_keys) & set(checkpoint_data.context_keys)
        or set(train_data.sample_keys) & set(checkpoint_data.sample_keys)
        or set(train_data.folds) & set(checkpoint_data.folds)
    ):
        raise ValueError("TriVUS train/checkpoint data must be disjoint")
    spec = model_spec(spec_id)
    weighted_train, standardizer = prepare_spec_data(train_data, spec)
    weighted_checkpoint = with_model_weights(
        checkpoint_data, spec["variant"], spec["target_family"]
    )
    train = standardizer.transform(weighted_train)
    checkpoint = standardizer.transform(weighted_checkpoint)
    set_seed(seed)
    model = make_model(train.features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, config["optimizer"]["maximum_epochs"] + 1):
        train_epoch(
            model, train, optimizer, seed, epoch,
            config["optimizer"]["batch_size"],
            config["optimizer"]["gradient_clip_norm"], device,
        )
        value = evaluate_loss(
            model, checkpoint,
            config["optimizer"]["evaluation_batch_size"], device,
        )
        history.append({"epoch": epoch, "checkpoint_loss": value})
        if value < best_loss - config["optimizer"]["minimum_improvement"]:
            best_loss = value
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= config["optimizer"]["patience"]:
            break
    if best_state is None or best_epoch < 1:
        raise ValueError("TriVUS checkpoint selection failed")
    model.load_state_dict(best_state)
    return model, standardizer, {
        "selected_epoch": best_epoch,
        "selected_checkpoint_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
    }


def train_fixed_epochs(train_data, spec_id, epochs, config, seed, device):
    if epochs < 1:
        raise ValueError("TriVUS fixed epochs must be positive")
    spec = model_spec(spec_id)
    weighted, standardizer = prepare_spec_data(train_data, spec)
    train = standardizer.transform(weighted)
    set_seed(seed)
    model = make_model(train.features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    for epoch in range(1, epochs + 1):
        train_epoch(
            model, train, optimizer, seed, epoch,
            config["optimizer"]["batch_size"],
            config["optimizer"]["gradient_clip_norm"], device,
        )
    return model, standardizer


def half_up_median(epochs):
    if len(epochs) != 4 or any(int(value) < 1 for value in epochs):
        raise ValueError("TriVUS final epoch requires four positive inner epochs")
    return max(1, int(math.floor(float(np.median(epochs)) + 0.5)))


def predict_data(model, data, standardizer, spec_id, batch_size, device):
    spec = model_spec(spec_id)
    if any(family not in spec["families"] for family in data.families):
        raise ValueError(f"TriVUS prediction data outside spec families: {spec_id}")
    weighted = with_model_weights(data, spec["variant"], spec["target_family"])
    transformed = standardizer.transform(weighted)
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(transformed), batch_size):
            indices = np.arange(start, min(start + batch_size, len(transformed)))
            batch = torch_batch(transformed, indices, device)
            utility, fallback_logit = model(
                batch.features, batch.candidate_mask, batch.fallback_indices
            )
            utility = utility.float().cpu().numpy()
            wrong = torch.sigmoid(-fallback_logit).float().cpu().numpy()
            for offset, index in enumerate(indices):
                valid = int(transformed.candidate_mask[index].sum())
                direct = int(np.argmax(utility[offset, :valid]))
                fallback = int(transformed.fallback_indices[index])
                labels = transformed.labels[index]
                output.append({
                    "context_key": transformed.context_keys[index],
                    "sample_key": transformed.sample_keys[index],
                    "family": transformed.families[index],
                    "cell": transformed.cells[index],
                    "row_id": transformed.row_ids[index],
                    "fold": int(transformed.folds[index]),
                    "group": transformed.groups[index],
                    "direct_index": direct,
                    "fallback_index": fallback,
                    "changed": direct != fallback,
                    "margin": float(utility[offset, direct] - utility[offset, 12]),
                    "wrong_score": float(wrong[offset]),
                    "direct_success": bool(labels[direct]),
                    "fallback_success": bool(labels[fallback]),
                })
    return output