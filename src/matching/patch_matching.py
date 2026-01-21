"""Patch-level matching between detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch


@dataclass
class PatchMatch:
    """A match between two patches."""
    idx1: int  # Index in first detection's valid patches
    idx2: int  # Index in second detection's valid patches
    coord1: Tuple[int, int]  # (row, col) in patch grid for detection 1
    coord2: Tuple[int, int]  # (row, col) in patch grid for detection 2
    similarity: float


def extract_valid_patches(
    features: torch.Tensor,
    patch_mask: torch.Tensor,
    pca: Optional[object] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extract valid patch features and their grid coordinates.

    Args:
        features: [C, H, W] patch features
        patch_mask: [H, W] boolean mask of valid patches
        pca: Optional PCA processor to reduce dimensionality

    Returns:
        patches: [N, D] array of patch features (N = number of valid patches)
        coords: List of (row, col) grid coordinates for each patch
    """
    C, H, W = features.shape

    # Get coordinates of valid patches
    coords = []
    for i in range(H):
        for j in range(W):
            if patch_mask[i, j]:
                coords.append((i, j))

    if not coords:
        return np.array([]).reshape(0, C), []

    # Extract features for valid patches
    patches = []
    for i, j in coords:
        patch_feat = features[:, i, j].cpu().numpy()
        patches.append(patch_feat)

    patches = np.array(patches)  # [N, C]

    # Apply PCA if provided
    if pca is not None and hasattr(pca, 'transform'):
        patches = pca.transform(patches)

    return patches, coords


def compute_patch_similarities(
    patches1: np.ndarray,
    patches2: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between all patch pairs.

    Args:
        patches1: [N1, D] features from first detection
        patches2: [N2, D] features from second detection

    Returns:
        similarities: [N1, N2] cosine similarity matrix
    """
    if patches1.shape[0] == 0 or patches2.shape[0] == 0:
        return np.array([]).reshape(patches1.shape[0], patches2.shape[0])

    # L2 normalize
    patches1_norm = patches1 / (np.linalg.norm(patches1, axis=1, keepdims=True) + 1e-8)
    patches2_norm = patches2 / (np.linalg.norm(patches2, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity
    similarities = patches1_norm @ patches2_norm.T

    return similarities


def find_mutual_nearest_neighbors(
    similarities: np.ndarray,
    coords1: List[Tuple[int, int]],
    coords2: List[Tuple[int, int]],
    threshold: float = 0.0,
) -> List[PatchMatch]:
    """
    Find mutual nearest neighbor matches between patches.

    A match is mutual if patch i's nearest neighbor is patch j,
    AND patch j's nearest neighbor is patch i.

    Args:
        similarities: [N1, N2] similarity matrix
        coords1: Grid coordinates for patches in detection 1
        coords2: Grid coordinates for patches in detection 2
        threshold: Minimum similarity for a valid match

    Returns:
        List of PatchMatch objects for mutual nearest neighbors
    """
    if similarities.size == 0:
        return []

    N1, N2 = similarities.shape

    # Find nearest neighbors in each direction
    nn_1to2 = np.argmax(similarities, axis=1)  # [N1] - for each patch in 1, best match in 2
    nn_2to1 = np.argmax(similarities, axis=0)  # [N2] - for each patch in 2, best match in 1

    matches = []
    for i in range(N1):
        j = nn_1to2[i]
        # Check if mutual
        if nn_2to1[j] == i:
            sim = similarities[i, j]
            if sim >= threshold:
                matches.append(PatchMatch(
                    idx1=i,
                    idx2=j,
                    coord1=coords1[i],
                    coord2=coords2[j],
                    similarity=float(sim),
                ))

    return matches


def find_matches_ratio_test(
    similarities: np.ndarray,
    coords1: List[Tuple[int, int]],
    coords2: List[Tuple[int, int]],
    ratio: float = 0.75,
) -> List[PatchMatch]:
    """
    Find matches using Lowe's ratio test.

    Accept match only if best_sim / second_best_sim > 1/ratio,
    i.e., the best match is significantly better than second best.

    Args:
        similarities: [N1, N2] similarity matrix
        coords1: Grid coordinates for patches in detection 1
        coords2: Grid coordinates for patches in detection 2
        ratio: Ratio threshold (default 0.75 means second_best must be < 0.75 * best)

    Returns:
        List of PatchMatch objects passing the ratio test
    """
    if similarities.size == 0 or similarities.shape[1] < 2:
        return []

    matches = []
    for i in range(similarities.shape[0]):
        sorted_idx = np.argsort(similarities[i])[::-1]
        best_j = sorted_idx[0]
        second_j = sorted_idx[1]

        best_sim = similarities[i, best_j]
        second_sim = similarities[i, second_j]

        # Ratio test: accept if second best is significantly worse
        if best_sim > 0 and second_sim / best_sim < ratio:
            matches.append(PatchMatch(
                idx1=i,
                idx2=int(best_j),
                coord1=coords1[i],
                coord2=coords2[best_j],
                similarity=float(best_sim),
            ))

    return matches


def find_top_k_matches(
    similarities: np.ndarray,
    coords1: List[Tuple[int, int]],
    coords2: List[Tuple[int, int]],
    k: int = 1,
    threshold: float = 0.0,
) -> List[PatchMatch]:
    """
    Find top-k matches for each patch in detection 1.

    Args:
        similarities: [N1, N2] similarity matrix
        coords1: Grid coordinates for patches in detection 1
        coords2: Grid coordinates for patches in detection 2
        k: Number of top matches per patch
        threshold: Minimum similarity for a valid match

    Returns:
        List of PatchMatch objects
    """
    if similarities.size == 0:
        return []

    N1, N2 = similarities.shape
    k = min(k, N2)

    matches = []
    for i in range(N1):
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        for j in top_k_indices:
            sim = similarities[i, j]
            if sim >= threshold:
                matches.append(PatchMatch(
                    idx1=i,
                    idx2=int(j),
                    coord1=coords1[i],
                    coord2=coords2[j],
                    similarity=float(sim),
                ))

    return matches
