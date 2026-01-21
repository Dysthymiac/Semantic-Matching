"""Re-identification accuracy metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ReIDMetrics:
    """Re-identification evaluation metrics."""
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    mean_reciprocal_rank: float
    num_queries: int
    num_gallery: int


def compute_reid_accuracy(
    fisher_vectors: Dict[str, np.ndarray],
    identity_mapping: Dict[str, str],
    exclude_same_image: bool = True,
    detection_to_image: Optional[Dict[str, str]] = None,
) -> ReIDMetrics:
    """
    Compute re-identification accuracy using Fisher Vector similarity.

    For each query detection, find the most similar detections and check
    if any of the top-k matches have the same identity.

    Args:
        fisher_vectors: Dict mapping detection_id -> Fisher Vector
        identity_mapping: Dict mapping detection_id -> individual_id
        exclude_same_image: If True, exclude matches from the same image
        detection_to_image: Dict mapping detection_id -> image_id (needed if exclude_same_image=True)

    Returns:
        ReIDMetrics with top-k accuracies and MRR
    """
    # Filter to detections that have both FV and identity
    valid_ids = [
        det_id for det_id in fisher_vectors.keys()
        if det_id in identity_mapping
    ]

    if len(valid_ids) < 2:
        return ReIDMetrics(
            top1_accuracy=0.0,
            top5_accuracy=0.0,
            top10_accuracy=0.0,
            mean_reciprocal_rank=0.0,
            num_queries=0,
            num_gallery=0
        )

    # Stack Fisher vectors and normalize
    det_ids = list(valid_ids)
    fvs = np.stack([fisher_vectors[det_id] for det_id in det_ids])
    fvs_norm = fvs / (np.linalg.norm(fvs, axis=1, keepdims=True) + 1e-8)

    # Compute similarity matrix
    similarity_matrix = fvs_norm @ fvs_norm.T

    # Metrics accumulators
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    reciprocal_ranks = []

    for i, query_id in enumerate(det_ids):
        query_identity = identity_mapping[query_id]
        query_image = detection_to_image.get(query_id) if detection_to_image else None

        # Get similarities to all other detections
        sims = similarity_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self

        # Optionally exclude same-image detections
        if exclude_same_image and detection_to_image:
            for j, other_id in enumerate(det_ids):
                if detection_to_image.get(other_id) == query_image:
                    sims[j] = -np.inf

        # Rank by similarity
        ranked_indices = np.argsort(sims)[::-1]

        # Find rank of first correct match
        first_correct_rank = None
        for rank, idx in enumerate(ranked_indices):
            if sims[idx] == -np.inf:
                continue
            other_identity = identity_mapping[det_ids[idx]]
            if other_identity == query_identity:
                first_correct_rank = rank + 1  # 1-indexed
                break

        if first_correct_rank is not None:
            reciprocal_ranks.append(1.0 / first_correct_rank)
            if first_correct_rank <= 1:
                top1_correct += 1
            if first_correct_rank <= 5:
                top5_correct += 1
            if first_correct_rank <= 10:
                top10_correct += 1
        else:
            reciprocal_ranks.append(0.0)

    num_queries = len(det_ids)

    return ReIDMetrics(
        top1_accuracy=top1_correct / num_queries if num_queries > 0 else 0.0,
        top5_accuracy=top5_correct / num_queries if num_queries > 0 else 0.0,
        top10_accuracy=top10_correct / num_queries if num_queries > 0 else 0.0,
        mean_reciprocal_rank=np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        num_queries=num_queries,
        num_gallery=num_queries  # Gallery = all other detections
    )
