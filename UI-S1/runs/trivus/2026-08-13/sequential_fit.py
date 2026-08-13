import random

import numpy as np
import torch

from candidate_success import candidate_success_loss
from sequential_model import SequentialCandidateVerifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(input_dimension, config):
    section = config["cheap_ranker"] if input_dimension == 115 else config["strong_verifier"]
    return SequentialCandidateVerifier(
        input_dimension=input_dimension,
        width=section.get("width", section.get("hidden_width")),
        heads=section.get("heads", 4),
        layers=section.get("layers", section.get("hidden_layers")),
        dropout=section["dropout"],
    )


def train_epoch(
    model, features, candidate_mask, labels, row_weights, optimizer,
    batch_size, pairwise_weight, gradient_clip_norm, seed,
):
    if len(features) != len(row_weights) or not bool(row_weights.sum() > 0):
        raise ValueError("Sequential training rows/weights mismatch")
    model.train()
    active = torch.nonzero(row_weights > 0, as_tuple=True)[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = active[
        torch.randperm(len(active), generator=generator).to(active.device)
    ]
    normalization = row_weights[active].sum().clamp_min(
        torch.finfo(row_weights.dtype).eps
    )
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, len(order), batch_size):
        selected = order[start:start + batch_size]
        logits, _ = model(features[selected], candidate_mask[selected])
        loss, _ = candidate_success_loss(
            logits,
            labels[selected],
            candidate_mask[selected],
            row_weights=row_weights[selected],
            pairwise_weight=pairwise_weight,
        )
        scaled = loss * (row_weights[selected].sum() / normalization)
        scaled.backward()
        total += float(scaled.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return total


def evaluate_loss(
    model, features, candidate_mask, labels, row_weights,
    batch_size, pairwise_weight,
):
    model.eval()
    total = 0.0
    active = torch.nonzero(row_weights > 0, as_tuple=True)[0]
    if not len(active):
        raise ValueError("Sequential evaluation has no positive-weight rows")
    normalization = row_weights[active].sum().clamp_min(
        torch.finfo(row_weights.dtype).eps
    )
    with torch.no_grad():
        for start in range(0, len(active), batch_size):
            selected = active[start:start + batch_size]
            logits, _ = model(features[selected], candidate_mask[selected])
            loss, _ = candidate_success_loss(
                logits,
                labels[selected],
                candidate_mask[selected],
                row_weights=row_weights[selected],
                pairwise_weight=pairwise_weight,
            )
            total += float(loss * (row_weights[selected].sum() / normalization))
    return total


def fit_with_checkpoint(
    train, checkpoint, input_dimension, config, seed, device,
):
    set_seed(seed)
    model = make_model(input_dimension, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    pairwise_weight = (
        config["cheap_ranker"]["pairwise_weight"]
        if input_dimension == 115
        else config["strong_verifier"]["pairwise_weight"]
    )
    best = None
    stale = 0
    history = []
    for epoch in range(1, config["optimizer"]["maximum_epochs"] + 1):
        train_epoch(
            model, *train, optimizer,
            config["optimizer"]["batch_size"], pairwise_weight,
            config["optimizer"]["gradient_clip_norm"], seed + epoch,
        )
        loss = evaluate_loss(
            model, *checkpoint,
            config["optimizer"]["evaluation_batch_size"], pairwise_weight,
        )
        history.append({"epoch": epoch, "checkpoint_loss": loss})
        if best is None or loss < best[0] - config["optimizer"]["minimum_improvement"]:
            best = (
                loss,
                epoch,
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            )
            stale = 0
        else:
            stale += 1
        if stale >= config["optimizer"]["patience"]:
            break
    if best is None:
        raise ValueError("Sequential checkpoint selection failed")
    model.load_state_dict(best[2])
    return model, {
        "selected_epoch": best[1],
        "selected_checkpoint_loss": best[0],
        "history": history,
    }


def require_real_data_optimizer_authorization(
    config, receipt=None, outer_fold=None, holdout_fold=None, family=None, phase=None,
):
    if config["execution"].get("real_data_optimizer_authorized") is not False:
        raise PermissionError("Sequential prereg authorization boundary mismatch")
    if receipt is None:
        raise PermissionError("Sequential real-data optimizer is not authorized")
    from sequential_authorization import validate_worker_receipt
    validate_worker_receipt(
        receipt, outer_fold, holdout_fold, family, phase
    )
    return True