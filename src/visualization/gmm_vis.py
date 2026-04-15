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

    # PCA transform (with band selection if configured) then GMM predict
    reduced = pca.transform_with_band(valid_features)
    return gmm.predict_proba(reduced)


def _generate_distinct_colors(n: int) -> np.ndarray:
    """Generate n maximally distinct colors using varied hue, saturation, and value."""
    colors = []

    # Base set of 20 hand-picked maximally distinct colors (Kelly colors + extras)
    base_colors = [
        [0.902, 0.098, 0.294],  # Red
        [0.235, 0.706, 0.294],  # Green
        [0.263, 0.388, 0.847],  # Blue
        [1.000, 0.882, 0.098],  # Yellow
        [0.957, 0.510, 0.188],  # Orange
        [0.569, 0.118, 0.706],  # Purple
        [0.275, 0.941, 0.941],  # Cyan
        [0.941, 0.196, 0.902],  # Magenta
        [0.824, 0.961, 0.235],  # Lime
        [0.980, 0.745, 0.831],  # Pink
        [0.000, 0.502, 0.502],  # Teal
        [0.902, 0.745, 1.000],  # Lavender
        [0.667, 0.431, 0.157],  # Brown
        [1.000, 0.980, 0.784],  # Beige
        [0.502, 0.000, 0.000],  # Maroon
        [0.667, 1.000, 0.765],  # Mint
        [0.502, 0.502, 0.000],  # Olive
        [1.000, 0.843, 0.706],  # Apricot
        [0.000, 0.000, 0.502],  # Navy
        [0.502, 0.502, 0.502],  # Grey
    ]

    if n <= len(base_colors):
        return np.array(base_colors[:n])

    # For more colors, generate using HSV with varied saturation and value
    colors = list(base_colors)

    # Generate additional colors with different saturation/value combinations
    sat_val_combos = [(1.0, 1.0), (0.7, 1.0), (1.0, 0.7), (0.5, 1.0), (1.0, 0.5)]
    combo_idx = 0
    hue = 0.0

    while len(colors) < n:
        sat, val = sat_val_combos[combo_idx % len(sat_val_combos)]

        # Convert HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append([r, g, b])

        # Golden ratio increment for hue
        hue = (hue + 0.618033988749895) % 1.0

        # Change sat/val combo every full hue cycle
        if hue < 0.618033988749895:
            combo_idx += 1

    return np.array(colors[:n])


def responsibilities_to_colors(
    responsibilities: np.ndarray,
    hard_assignment: bool = True
) -> np.ndarray:
    """
    Convert GMM responsibilities to RGB colors.

    Args:
        responsibilities: [N_patches, n_components]
        hard_assignment: If True, use argmax (most likely component).
                        If False, blend colors by responsibility weights.

    Returns:
        colors: [N_patches, 3] RGB values in [0, 1]
    """
    n_patches, n_components = responsibilities.shape

    # Generate maximally distinct colors
    component_colors = _generate_distinct_colors(n_components)

    if hard_assignment:
        # Hard assignment: each patch gets the color of its most likely component
        assignments = np.argmax(responsibilities, axis=1)
        return component_colors[assignments]
    else:
        # Soft assignment: blend colors by responsibilities
        return np.clip(responsibilities @ component_colors, 0, 1)
