"""Fisher Vector encoding utilities using scikit-image."""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from skimage.feature import fisher_vector
from sklearn.mixture import GaussianMixture

from ..data.preprocessed_dataset import Detection


def encode_detection_fisher_vector(
    detection: Detection,
    gmm: GaussianMixture,
    pca_processor=None
) -> Optional[np.ndarray]:
    """
    Encode a single detection as a Fisher Vector.

    Args:
        detection: Detection with features and patch_mask
        gmm: Trained Gaussian Mixture Model
        pca_processor: Optional PCA processor to transform features

    Returns:
        Fisher Vector encoding or None if no valid patches
    """
    if detection.features.numel() == 0 or detection.patch_mask.numel() == 0:
        return None

    # Extract valid patches
    embed_dim, h_patches, w_patches = detection.features.shape
    features_flat = detection.features.view(embed_dim, -1)  # [embed_dim, num_patches]
    patch_mask_flat = detection.patch_mask.flatten().bool()  # [num_patches]

    # Select only valid patches
    valid_features = features_flat[:, patch_mask_flat].T.cpu().numpy()  # [valid_patches, embed_dim]

    if valid_features.shape[0] == 0:
        return None

    # Apply PCA if provided
    if pca_processor is not None:
        valid_features = pca_processor.pca.transform(valid_features).astype(np.float32)

    # Compute Fisher Vector using skimage with improved normalization
    # improved=True applies power normalization (alpha=0.5) and L2 normalization
    fv = fisher_vector(valid_features, gmm, improved=True)

    return fv.astype(np.float32)


def compute_fisher_vector_stats(fv: np.ndarray) -> dict:
    """
    Compute statistics for a Fisher Vector.

    Args:
        fv: Fisher Vector encoding

    Returns:
        Dictionary of statistics
    """
    return {
        'dim': fv.shape[0],
        'norm': np.linalg.norm(fv),
        'min': float(fv.min()),
        'max': float(fv.max()),
        'mean': float(fv.mean()),
        'std': float(fv.std()),
        'sparsity': float((fv == 0).mean())
    }