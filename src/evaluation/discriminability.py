"""Discriminability metrics for evaluating feature quality."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform


def fisher_discriminant_1d(values: np.ndarray, labels: np.ndarray) -> float:
    """Compute the 1D Fisher discriminant ratio (between / within class variance).

    Higher ratio means the feature separates the classes better.

    Args:
        values: [N] feature values.
        labels: [N] integer or string class labels.

    Returns:
        Fisher ratio (between-class variance / within-class variance).
    """
    overall_mean = values.mean()
    unique_labels = np.unique(labels)
    between = 0.0
    within = 0.0
    for ul in unique_labels:
        m = labels == ul
        if m.sum() < 2:
            continue
        cls_mean = values[m].mean()
        between += m.sum() * (cls_mean - overall_mean) ** 2
        within += ((values[m] - cls_mean) ** 2).sum()
    return between / max(within, 1e-20)


def pairwise_same_class_auc(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subsample: int = 3000,
    metric: str = "euclidean",
    seed: int = 42,
) -> float:
    """Compute AUC for same-class pair discrimination from an embedding.

    Subsamples to avoid O(N^2) blowup. Uses negative distance as
    the similarity score.

    Args:
        embeddings: [N, d] embedding matrix.
        labels: [N] class labels (string or int).
        subsample: Max number of points to use.
        metric: Distance metric for pdist.
        seed: Random seed for subsampling.

    Returns:
        AUC score (higher = better class separation).
    """
    N = len(embeddings)
    if N > subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, subsample, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    triu = np.triu_indices(len(embeddings), k=1)
    same_class = labels[triu[0]] == labels[triu[1]]

    if same_class.all() or (~same_class).all():
        return 0.5  # degenerate: only one class present

    dists = squareform(pdist(embeddings, metric))
    sim = -dists[triu]
    return float(roc_auc_score(same_class, sim))
