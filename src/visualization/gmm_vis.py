"""GMM responsibility visualization utilities."""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from ..pca.incremental_pca import IncrementalPCAProcessor


def compute_patch_responsibilities(
    features: torch.Tensor,
    patch_mask: torch.Tensor,
    pca: IncrementalPCAProcessor,
    gmm: GaussianMixture
) -> np.ndarray:
    """
    Compute GMM responsibilities for valid patches.

    Args:
        features: [embed_dim, H_patches, W_patches]
        patch_mask: [H_patches, W_patches] boolean
        pca: Fitted PCA processor
        gmm: Fitted GMM model

    Returns:
        responsibilities: [N_valid_patches, n_components]
    """
    # Extract valid patch features
    embed_dim, h, w = features.shape
    features_flat = features.reshape(embed_dim, -1).T  # [H*W, embed_dim]
    mask_flat = patch_mask.flatten().bool()
    valid_features = features_flat[mask_flat].numpy()  # [N_valid, embed_dim]

    if len(valid_features) == 0:
        return np.empty((0, gmm.n_components))

    # PCA transform then GMM predict
    reduced = pca.pca.transform(valid_features)
    return gmm.predict_proba(reduced)


def responsibilities_to_colors(
    responsibilities: np.ndarray,
    cmap_name: str = 'tab20'
) -> np.ndarray:
    """
    Convert GMM responsibilities to RGB colors.

    Uses soft assignment: blends component colors by responsibility weights.

    Args:
        responsibilities: [N_patches, n_components]
        cmap_name: Matplotlib colormap name

    Returns:
        colors: [N_patches, 3] RGB values in [0, 1]
    """
    n_patches, n_components = responsibilities.shape
    cmap = plt.get_cmap(cmap_name)

    # Get color for each component
    component_colors = np.array([cmap(i / n_components)[:3] for i in range(n_components)])

    # Blend colors by responsibilities (clip to ensure valid RGB range)
    return np.clip(responsibilities @ component_colors, 0, 1)
