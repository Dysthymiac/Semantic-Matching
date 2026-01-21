"""Diagnostic for evaluating whether texture features encode identity information.

This module provides a systematic test to determine if texture features can distinguish
same-identity pairs from different-identity pairs when controlling for pose.

The key insight: if texture features encode identity, then among pose-matched detections
(similar semantic features), same-identity pairs should have higher texture similarity
than different-identity pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ..data.preprocessed_dataset import PreprocessedDataset
from ..pca.incremental_pca import IncrementalPCAProcessor
from ..utils.batch_storage import load_batch_from_file


@dataclass
class TextureDiagnosticResult:
    """Results from texture identity diagnostic."""
    # Core metric
    roc_auc: float  # 0.5 = random, 1.0 = perfect separation

    # Sample counts
    n_queries: int
    n_same_identity_pairs: int
    n_diff_identity_pairs: int

    # Similarity distributions
    same_identity_similarities: np.ndarray
    diff_identity_similarities: np.ndarray

    # Summary statistics
    same_identity_mean: float
    same_identity_std: float
    diff_identity_mean: float
    diff_identity_std: float

    def print_summary(self) -> None:
        """Print diagnostic summary."""
        print("=" * 60)
        print("TEXTURE IDENTITY DIAGNOSTIC RESULTS")
        print("=" * 60)
        print(f"Queries evaluated: {self.n_queries}")
        print(f"Same-identity pairs: {self.n_same_identity_pairs}")
        print(f"Different-identity pairs: {self.n_diff_identity_pairs}")
        print()
        print(f"Same-identity similarity:      {self.same_identity_mean:.4f} +/- {self.same_identity_std:.4f}")
        print(f"Different-identity similarity: {self.diff_identity_mean:.4f} +/- {self.diff_identity_std:.4f}")
        print()
        print(f"ROC-AUC: {self.roc_auc:.4f}")
        print()
        if self.roc_auc > 0.7:
            print("VERDICT: Texture features ENCODE identity (AUC > 0.7)")
        elif self.roc_auc > 0.55:
            print("VERDICT: Texture features have WEAK identity signal (0.55 < AUC < 0.7)")
        else:
            print("VERDICT: Texture features DO NOT encode identity (AUC <= 0.55)")
        print("=" * 60)


def pool_features(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pool patch features to detection-level vector using mean over valid patches.

    Args:
        features: [D, H, W] feature tensor
        mask: [H, W] boolean mask of valid patches

    Returns:
        [D] pooled feature vector, L2-normalized
    """
    # Flatten spatial dims: [D, H*W]
    D = features.shape[0]
    features_flat = features.reshape(D, -1)
    mask_flat = mask.flatten()

    # Mean over valid patches
    valid_features = features_flat[:, mask_flat]
    if valid_features.shape[1] == 0:
        return np.zeros(D)

    pooled = valid_features.mean(axis=1)

    # L2 normalize
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm

    return pooled


def load_detection_features(
    dataset: PreprocessedDataset,
    pca: Optional[IncrementalPCAProcessor] = None,
    max_detections: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load and pool features for all detections.

    Args:
        dataset: Preprocessed dataset
        pca: Optional PCA processor for dimensionality reduction
        max_detections: Optional limit on number of detections to load

    Returns:
        Tuple of (detection_id -> pooled_features, detection_id -> raw_features_for_patch_analysis)
    """
    pooled_features = {}

    batch_files = list(dataset._index['batch_to_detections'].keys())
    total_loaded = 0

    for batch_rel_path in tqdm(batch_files, desc="Loading features"):
        if max_detections and total_loaded >= max_detections:
            break

        batch_path = dataset.output_root / batch_rel_path
        if not batch_path.exists():
            continue

        batch_data = load_batch_from_file(batch_path)
        if not batch_data:
            continue

        for det_id, det in batch_data.items():
            if max_detections and total_loaded >= max_detections:
                break

            # Get features and mask as numpy
            features = det.features.numpy() if hasattr(det.features, 'numpy') else det.features
            mask = det.patch_mask.numpy() if hasattr(det.patch_mask, 'numpy') else det.patch_mask
            mask = mask.astype(bool)

            # Apply PCA if provided (transform each patch)
            if pca is not None and pca.is_fitted():
                D, H, W = features.shape
                features_flat = features.reshape(D, -1).T  # [H*W, D]
                features_transformed = pca.pca.transform(features_flat)  # [H*W, D_pca]
                features = features_transformed.T.reshape(-1, H, W)  # [D_pca, H, W]

            # Pool to detection-level
            pooled = pool_features(features, mask)
            pooled_features[det_id] = pooled
            total_loaded += 1

        del batch_data

    return pooled_features


def compute_texture_identity_separation(
    semantic_features: Dict[str, np.ndarray],
    textural_features: Dict[str, np.ndarray],
    identity_map: Dict[str, str],
    n_queries: int = 1000,
    k_pose_neighbors: int = 50,
    skip_top_k: int = 0,
    min_same_identity: int = 1,
    random_seed: int = 42,
) -> TextureDiagnosticResult:
    """
    Evaluate whether texture features encode identity by comparing similarity
    distributions for same-identity vs different-identity pairs within pose-matched sets.

    Algorithm:
    1. For each query detection, find K nearest neighbors by semantic similarity (pose-matched)
    2. Optionally skip top-k most similar (likely same-encounter matches)
    3. Within pose-matched set, compute texture similarity to query
    4. Label pairs as same-identity or different-identity
    5. Compute ROC-AUC for distinguishing same from different

    Args:
        semantic_features: Dict mapping detection_id -> pooled semantic features
        textural_features: Dict mapping detection_id -> pooled textural features
        identity_map: Dict mapping detection_id -> individual_id
        n_queries: Number of query detections to evaluate
        k_pose_neighbors: Number of pose-matched neighbors to consider
        skip_top_k: Skip the top-k most similar matches (likely same-encounter)
        min_same_identity: Minimum same-identity neighbors required to include query
        random_seed: Random seed for query sampling

    Returns:
        TextureDiagnosticResult with ROC-AUC and distributions
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Find detections that exist in all three mappings
    valid_ids = [
        det_id for det_id in semantic_features.keys()
        if det_id in textural_features and det_id in identity_map
    ]

    if len(valid_ids) < k_pose_neighbors + 1:
        raise ValueError(f"Not enough valid detections: {len(valid_ids)} < {k_pose_neighbors + 1}")

    # Stack features for efficient similarity computation
    det_ids = list(valid_ids)
    det_id_to_idx = {det_id: i for i, det_id in enumerate(det_ids)}

    semantic_matrix = np.stack([semantic_features[det_id] for det_id in det_ids])
    textural_matrix = np.stack([textural_features[det_id] for det_id in det_ids])

    # Normalize for cosine similarity
    semantic_matrix = semantic_matrix / (np.linalg.norm(semantic_matrix, axis=1, keepdims=True) + 1e-8)
    textural_matrix = textural_matrix / (np.linalg.norm(textural_matrix, axis=1, keepdims=True) + 1e-8)

    # Precompute semantic similarities
    semantic_sim_matrix = semantic_matrix @ semantic_matrix.T

    # Sample queries
    query_ids = random.sample(det_ids, min(n_queries, len(det_ids)))

    # Collect similarity pairs
    same_identity_sims = []
    diff_identity_sims = []
    queries_used = 0

    for query_id in tqdm(query_ids, desc="Evaluating queries"):
        query_idx = det_id_to_idx[query_id]
        query_identity = identity_map[query_id]

        # Find K nearest neighbors by semantic similarity (excluding self)
        semantic_sims = semantic_sim_matrix[query_idx].copy()
        semantic_sims[query_idx] = -np.inf  # Exclude self

        # Get top (skip_top_k + k_pose_neighbors) and then skip the first skip_top_k
        all_neighbor_indices = np.argsort(semantic_sims)[::-1][:skip_top_k + k_pose_neighbors]
        neighbor_indices = all_neighbor_indices[skip_top_k:]  # Skip top-k (likely same-encounter)

        # Check if we have any same-identity neighbors in pose-matched set
        same_id_in_neighbors = 0
        for neighbor_idx in neighbor_indices:
            neighbor_id = det_ids[neighbor_idx]
            if identity_map[neighbor_id] == query_identity:
                same_id_in_neighbors += 1

        if same_id_in_neighbors < min_same_identity:
            continue  # Skip queries without same-identity pose-matched neighbors

        queries_used += 1

        # Compute texture similarity to query for all pose-matched neighbors
        query_texture = textural_matrix[query_idx]

        for neighbor_idx in neighbor_indices:
            neighbor_id = det_ids[neighbor_idx]
            neighbor_texture = textural_matrix[neighbor_idx]

            texture_sim = float(query_texture @ neighbor_texture)

            if identity_map[neighbor_id] == query_identity:
                same_identity_sims.append(texture_sim)
            else:
                diff_identity_sims.append(texture_sim)

    # Convert to arrays
    same_identity_sims = np.array(same_identity_sims)
    diff_identity_sims = np.array(diff_identity_sims)

    if len(same_identity_sims) == 0 or len(diff_identity_sims) == 0:
        raise ValueError("No valid pairs found. Check identity_map coverage.")

    # Compute ROC-AUC
    # Labels: 1 = same identity, 0 = different identity
    labels = np.concatenate([
        np.ones(len(same_identity_sims)),
        np.zeros(len(diff_identity_sims))
    ])
    scores = np.concatenate([same_identity_sims, diff_identity_sims])

    roc_auc = roc_auc_score(labels, scores)

    return TextureDiagnosticResult(
        roc_auc=roc_auc,
        n_queries=queries_used,
        n_same_identity_pairs=len(same_identity_sims),
        n_diff_identity_pairs=len(diff_identity_sims),
        same_identity_similarities=same_identity_sims,
        diff_identity_similarities=diff_identity_sims,
        same_identity_mean=float(np.mean(same_identity_sims)),
        same_identity_std=float(np.std(same_identity_sims)),
        diff_identity_mean=float(np.mean(diff_identity_sims)),
        diff_identity_std=float(np.std(diff_identity_sims)),
    )


def run_texture_diagnostic(
    semantic_dataset: PreprocessedDataset,
    textural_dataset: PreprocessedDataset,
    identity_map: Dict[str, str],
    semantic_pca: Optional[IncrementalPCAProcessor] = None,
    textural_pca: Optional[IncrementalPCAProcessor] = None,
    n_queries: int = 1000,
    k_pose_neighbors: int = 50,
    skip_top_k: int = 0,
    max_detections: Optional[int] = None,
) -> TextureDiagnosticResult:
    """
    Convenience function to run the full texture diagnostic pipeline.

    Args:
        semantic_dataset: Dataset with semantic features (e.g., DINO)
        textural_dataset: Dataset with textural features (e.g., SIFT)
        identity_map: Dict mapping detection_id -> individual_id
        semantic_pca: Optional PCA for semantic features
        textural_pca: Optional PCA for textural features
        n_queries: Number of queries to evaluate
        k_pose_neighbors: Number of pose-matched neighbors
        skip_top_k: Skip top-k most similar matches (likely same-encounter)
        max_detections: Optional limit on detections to load

    Returns:
        TextureDiagnosticResult
    """
    print("Loading semantic features...")
    semantic_features = load_detection_features(
        semantic_dataset, pca=semantic_pca, max_detections=max_detections
    )
    print(f"Loaded {len(semantic_features)} semantic feature vectors")

    print("\nLoading textural features...")
    textural_features = load_detection_features(
        textural_dataset, pca=textural_pca, max_detections=max_detections
    )
    print(f"Loaded {len(textural_features)} textural feature vectors")

    print(f"\nRunning diagnostic (skip_top_k={skip_top_k})...")
    result = compute_texture_identity_separation(
        semantic_features=semantic_features,
        textural_features=textural_features,
        identity_map=identity_map,
        n_queries=n_queries,
        k_pose_neighbors=k_pose_neighbors,
        skip_top_k=skip_top_k,
    )

    return result


def plot_diagnostic_distributions(
    result: TextureDiagnosticResult,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Plot similarity distributions for same-identity vs different-identity pairs.

    Args:
        result: TextureDiagnosticResult from diagnostic
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax = axes[0]
    ax.hist(result.same_identity_similarities, bins=50, alpha=0.7,
            label=f'Same identity (n={result.n_same_identity_pairs})', color='green')
    ax.hist(result.diff_identity_similarities, bins=50, alpha=0.7,
            label=f'Different identity (n={result.n_diff_identity_pairs})', color='red')
    ax.axvline(result.same_identity_mean, color='green', linestyle='--', linewidth=2)
    ax.axvline(result.diff_identity_mean, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Texture Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title(f'Texture Similarity Distributions\n(ROC-AUC = {result.roc_auc:.4f})')
    ax.legend()

    # Box plot
    ax = axes[1]
    data = [result.same_identity_similarities, result.diff_identity_similarities]
    bp = ax.boxplot(data, labels=['Same Identity', 'Different Identity'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.7)
    ax.set_ylabel('Texture Cosine Similarity')
    ax.set_title('Distribution Comparison')

    plt.tight_layout()
    plt.show()
