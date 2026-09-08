"""Optional differentiable regularization for a trained OPQ/PQ codebook.

Faiss OPQ optimizes quantization distortion, but it does not explicitly
encourage balanced code usage or discourage redundant product coordinates.
This module keeps the learned OPQ rotation fixed and fine-tunes only the PQ
centroids with three transparent terms:

* soft reconstruction error;
* KL divergence between each coordinate's aggregate posterior and uniform;
* mutual information between aggregate posteriors of coordinate pairs.

Hard codes and collision repair remain the responsibility of the tokenizer.
The feature is opt-in so canonical OPQ artifacts are unchanged.
"""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import torch
import torch.nn.functional as F


def _soft_assignments(inputs, centroids, temperature):
    """Return squared distances and soft assignments for product subvectors."""
    distances = (
        inputs.square().sum(dim=-1, keepdim=True)
        + centroids.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * torch.einsum("bdm,dkm->bdk", inputs, centroids)
    ).clamp_min(0.0)
    assignments = F.softmax(-distances / max(float(temperature), 1e-6), dim=-1)
    return distances, assignments


def regularized_pq_loss(
    inputs,
    centroids,
    initial_centroids,
    temperature,
    balance_weight,
    mi_weight,
    hardness_weight,
    anchor_weight,
    straight_through=False,
):
    """Compute reconstruction, usage-balance, and coordinate-MI losses."""
    _, soft_assignments = _soft_assignments(inputs, centroids, temperature)
    if straight_through:
        hard = F.one_hot(
            soft_assignments.argmax(dim=-1),
            num_classes=soft_assignments.shape[-1],
        ).to(soft_assignments.dtype)
        assignments = hard - soft_assignments.detach() + soft_assignments
    else:
        assignments = soft_assignments
    reconstruction = torch.einsum("bdk,dkm->bdm", assignments, centroids)
    reconstruction_loss = F.mse_loss(reconstruction, inputs)

    eps = torch.finfo(assignments.dtype).eps
    marginals = assignments.mean(dim=0).clamp_min(eps)
    balance_loss = (
        marginals * (marginals.log() + math.log(assignments.shape[-1]))
    ).sum(dim=-1).mean()
    assignment_entropy = -(
        assignments.clamp_min(eps) * assignments.clamp_min(eps).log()
    ).sum(dim=-1).mean()

    pair_losses = []
    for left, right in combinations(range(assignments.shape[1]), 2):
        joint = torch.einsum(
            "bk,bl->kl", assignments[:, left], assignments[:, right]
        ) / assignments.shape[0]
        joint = joint.clamp_min(eps)
        independent = (
            marginals[left][:, None] * marginals[right][None, :]
        ).clamp_min(eps)
        pair_losses.append((joint * (joint.log() - independent.log())).sum())
    mi_loss = (
        torch.stack(pair_losses).mean()
        if pair_losses
        else reconstruction_loss.new_zeros(())
    )
    anchor_loss = F.mse_loss(centroids, initial_centroids)
    total = (
        reconstruction_loss
        + float(balance_weight) * balance_loss
        + float(mi_weight) * mi_loss
        + float(hardness_weight) * assignment_entropy
        + float(anchor_weight) * anchor_loss
    )
    metrics = {
        "loss": float(total.detach()),
        "reconstruction": float(reconstruction_loss.detach()),
        "balance_kl": float(balance_loss.detach()),
        "soft_pairwise_mi": float(mi_loss.detach()),
        "assignment_entropy": float(assignment_entropy.detach()),
        "centroid_anchor": float(anchor_loss.detach()),
    }
    return total, metrics


def hard_product_codes(inputs: np.ndarray, centroids: np.ndarray):
    """Assign every product subvector to its nearest centroid."""
    distances = (
        np.square(inputs).sum(axis=-1, keepdims=True)
        + np.square(centroids).sum(axis=-1)[None, :, :]
        - 2.0 * np.einsum("ndm,dkm->ndk", inputs, centroids)
    )
    return distances.argmin(axis=-1).astype(np.int64)


def hard_quantization_distortion(inputs, centroids, codes):
    """Mean squared product-vector reconstruction error under hard codes."""
    digit_rows = np.arange(inputs.shape[1])[None, :]
    reconstruction = centroids[digit_rows, codes]
    return float(np.square(inputs - reconstruction).sum(axis=(1, 2)).mean())


def regularize_product_centroids(
    pq_inputs: np.ndarray,
    initial_centroids: np.ndarray,
    train_mask: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    balance_weight: float,
    mi_weight: float,
    hardness_weight: float,
    anchor_weight: float,
    device: str,
    seed: int,
    straight_through: bool = False,
):
    """Fine-tune PQ centroids and return hard codes plus a training report."""
    pq_inputs = np.asarray(pq_inputs, dtype=np.float32)
    initial_centroids = np.asarray(initial_centroids, dtype=np.float32)
    train_indices = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    if not len(train_indices):
        raise ValueError("OPQ regularization requires at least one training item")
    if pq_inputs.shape[1:] != initial_centroids.shape[::2]:
        expected = (initial_centroids.shape[0], initial_centroids.shape[2])
        raise ValueError(f"PQ input shape {pq_inputs.shape[1:]} != {expected}")

    torch_device = torch.device(device)
    initial = torch.from_numpy(initial_centroids).to(torch_device)
    centroids = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([centroids], lr=float(learning_rate))
    generator = np.random.default_rng(int(seed))
    history = []
    steps = int(steps)
    for step in range(steps):
        replace = len(train_indices) < int(batch_size)
        rows = generator.choice(
            train_indices, size=min(int(batch_size), len(train_indices)), replace=replace
        )
        batch = torch.from_numpy(pq_inputs[rows]).to(torch_device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = regularized_pq_loss(
            batch,
            centroids,
            initial,
            temperature=temperature,
            balance_weight=balance_weight,
            mi_weight=mi_weight,
            hardness_weight=hardness_weight,
            anchor_weight=anchor_weight,
            straight_through=straight_through,
        )
        loss.backward()
        optimizer.step()
        if step == 0 or step + 1 == steps or (step + 1) % max(1, steps // 10) == 0:
            history.append({"step": step + 1, **metrics})

    initial_codes = hard_product_codes(pq_inputs, initial_centroids)
    learned = centroids.detach().cpu().numpy()
    hard_codes = hard_product_codes(pq_inputs, learned)
    report = {
        "steps": steps,
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "balance_weight": float(balance_weight),
        "mi_weight": float(mi_weight),
        "hardness_weight": float(hardness_weight),
        "anchor_weight": float(anchor_weight),
        "device": str(torch_device),
        "seed": int(seed),
        "straight_through": bool(straight_through),
        "history": history,
        "changed_nearest_code_rate": float(
            np.any(hard_codes != initial_codes, axis=1).mean()
        ),
        "initial_hard_distortion": hard_quantization_distortion(
            pq_inputs, initial_centroids, initial_codes
        ),
        "regularized_hard_distortion": hard_quantization_distortion(
            pq_inputs, learned, hard_codes
        ),
        "centroid_rms_drift": float(
            np.sqrt(np.square(learned - initial_centroids).mean())
        ),
        "changed_native_code_rate": None,
    }
    return hard_codes, learned, report
