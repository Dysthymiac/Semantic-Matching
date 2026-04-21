"""Fisher Vector similarity decomposition and explanation visualization.

Decomposes Fisher Vector similarity by GMM component to show which
"body parts" contributed most to the match between two images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from PIL import Image

from .primitives import get_crop_bounds, patch_coords_in_crop
from .gmm_vis import compute_patch_responsibilities


@dataclass
class SimilarityDecomposition:
    """Result of decomposing Fisher Vector similarity by GMM component."""

    total_similarity: float
    component_contributions: np.ndarray  # [K] per-component contribution
    weight_contributions: np.ndarray     # [K] contribution from weight gradients
    mean_contributions: np.ndarray       # [K] contribution from mean gradients
    var_contributions: np.ndarray        # [K] contribution from variance gradients
    n_components: int
    feature_dim: int


def decompose_fisher_similarity(
    fv1: np.ndarray,
    fv2: np.ndarray,
    n_components: int,
    feature_dim: int,
) -> SimilarityDecomposition:
    """
    Decompose Fisher Vector similarity into per-component contributions.

    Fisher Vector structure (from skimage.feature.fisher_vector):
        FV = [∇π_1, ..., ∇π_K, ∇μ_1, ..., ∇μ_K, ∇σ_1, ..., ∇σ_K]
        where:
        - ∇π_k is a scalar (weight gradient for component k)
        - ∇μ_k has `feature_dim` dimensions (mean gradient)
        - ∇σ_k has `feature_dim` dimensions (variance gradient)

    Total dimension: K + K*D + K*D = K * (1 + 2*D)

    Total similarity = Σ_k (∇π_k^1 · ∇π_k^2 + ∇μ_k^1 · ∇μ_k^2 + ∇σ_k^1 · ∇σ_k^2)

    Args:
        fv1: First Fisher vector [K * (1 + 2*D)]
        fv2: Second Fisher vector [K * (1 + 2*D)]
        n_components: Number of GMM components (K)
        feature_dim: Feature dimension after PCA (D)

    Returns:
        SimilarityDecomposition with per-component contributions
    """
    K, D = n_components, feature_dim
    expected_dim = K * (1 + 2 * D)

    if fv1.shape[0] != expected_dim or fv2.shape[0] != expected_dim:
        raise ValueError(
            f"FV dimension mismatch: expected {expected_dim} = K*(1+2D) "
            f"where K={K}, D={D}, got {fv1.shape[0]} and {fv2.shape[0]}"
        )

    weight_contributions = np.zeros(K, dtype=np.float32)
    mean_contributions = np.zeros(K, dtype=np.float32)
    var_contributions = np.zeros(K, dtype=np.float32)

    # Weight gradient indices: [0 : K]
    for k in range(K):
        weight_contributions[k] = fv1[k] * fv2[k]

    # Mean gradient indices: [K + k*D : K + (k+1)*D]
    mean_offset = K
    for k in range(K):
        mean_start = mean_offset + k * D
        mean_end = mean_offset + (k + 1) * D
        mean_contributions[k] = np.dot(fv1[mean_start:mean_end], fv2[mean_start:mean_end])

    # Variance gradient indices: [K + K*D + k*D : K + K*D + (k+1)*D]
    var_offset = K + K * D
    for k in range(K):
        var_start = var_offset + k * D
        var_end = var_offset + (k + 1) * D
        var_contributions[k] = np.dot(fv1[var_start:var_end], fv2[var_start:var_end])

    component_contributions = weight_contributions + mean_contributions + var_contributions
    total_similarity = component_contributions.sum()

    return SimilarityDecomposition(
        total_similarity=float(total_similarity),
        component_contributions=component_contributions,
        weight_contributions=weight_contributions,
        mean_contributions=mean_contributions,
        var_contributions=var_contributions,
        n_components=K,
        feature_dim=D,
    )


def compute_patch_contribution_map(
    responsibilities: np.ndarray,
    component_contributions: np.ndarray,
) -> np.ndarray:
    """
    Map component contributions back to patches using responsibilities.

    For each patch, its contribution to the similarity is:
        patch_contribution = Σ_k (responsibility_k × contribution_k)

    Args:
        responsibilities: [N_patches, K] GMM responsibilities for each patch
        component_contributions: [K] per-component contribution to similarity

    Returns:
        patch_contributions: [N_patches] contribution of each patch
    """
    # Weight responsibilities by component contributions
    # [N_patches, K] @ [K] -> [N_patches]
    return responsibilities @ component_contributions


def get_top_contributing_components(
    decomposition: SimilarityDecomposition,
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get indices and values of top contributing components.

    Args:
        decomposition: Similarity decomposition result
        top_k: Number of top components to return

    Returns:
        (indices, contributions) of top contributing components
    """
    sorted_indices = np.argsort(decomposition.component_contributions)[::-1]
    top_indices = sorted_indices[:top_k]
    top_contributions = decomposition.component_contributions[top_indices]
    return top_indices, top_contributions


def visualize_similarity_explanation(
    det1,
    det2,
    fv1: np.ndarray,
    fv2: np.ndarray,
    pca_processor,
    gmm,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    cmap: str = 'RdYlGn',
    show_top_components: int = 5,
) -> plt.Figure:
    """
    Visualize which patches contributed most to Fisher Vector similarity.

    Shows both detections side-by-side with heatmaps indicating how much
    each patch contributed to the match. Also shows a bar chart of
    per-component contributions.

    Args:
        det1: First Detection object
        det2: Second Detection object (the match)
        fv1: Original Fisher vector for det1 (pre-PCA reduction)
        fv2: Original Fisher vector for det2
        pca_processor: PCA processor for feature transformation
        gmm: Fitted GMM model
        title: Optional title for the figure
        figsize: Figure size
        cmap: Colormap for contribution heatmap
        show_top_components: Number of top components to show in bar chart

    Returns:
        matplotlib Figure
    """
    n_components = gmm.n_components
    feature_dim = pca_processor.config.n_components

    # Decompose similarity
    decomp = decompose_fisher_similarity(fv1, fv2, n_components, feature_dim)

    # Compute responsibilities for both detections
    resp1 = compute_patch_responsibilities(det1.features, det1.patch_mask, pca_processor, gmm)
    resp2 = compute_patch_responsibilities(det2.features, det2.patch_mask, pca_processor, gmm)

    # Compute patch contribution maps
    contrib1 = compute_patch_contribution_map(resp1, decomp.component_contributions)
    contrib2 = compute_patch_contribution_map(resp2, decomp.component_contributions)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Common normalization for both heatmaps
    all_contribs = np.concatenate([contrib1, contrib2])
    vmin, vmax = all_contribs.min(), all_contribs.max()

    # Center colormap around zero if contributions span negative and positive
    if vmin < 0 and vmax > 0:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    # Helper to draw detection with contribution heatmap
    def draw_detection_heatmap(ax, det, contributions, label):
        img = Image.open(det.image_path).convert('RGB')
        img_w, img_h = img.size
        x1, y1, x2, y2 = get_crop_bounds(det.square_crop_bbox, img_w, img_h)
        crop = img.crop((x1, y1, x2, y2))
        crop_w, crop_h = x2 - x1, y2 - y1

        ax.imshow(crop)

        # Get patch coordinates
        coords = patch_coords_in_crop(det.patch_mask, crop_w, crop_h)

        # Draw patches with contribution-based colors
        for idx, (px, py, pw, ph) in enumerate(coords):
            color = cmap_obj(norm(contributions[idx]))
            ax.add_patch(Rectangle(
                (px, py), pw, ph,
                facecolor=color,
                edgecolor='none',
                alpha=0.7,
            ))

        ax.set_title(label, fontsize=10)
        ax.axis('off')

    # Draw both detections
    draw_detection_heatmap(axes[0], det1, contrib1, "Query")
    draw_detection_heatmap(axes[1], det2, contrib2, f"Match (sim={decomp.total_similarity:.3f})")

    # Bar chart of top component contributions
    ax_bar = axes[2]
    top_idx, top_vals = get_top_contributing_components(decomp, show_top_components)

    colors = ['green' if v > 0 else 'red' for v in top_vals]
    bars = ax_bar.barh(range(len(top_idx)), top_vals, color=colors, alpha=0.7)
    ax_bar.set_yticks(range(len(top_idx)))
    ax_bar.set_yticklabels([f'Comp {i}' for i in top_idx])
    ax_bar.set_xlabel('Contribution to similarity')
    ax_bar.set_title(f'Top {show_top_components} components')
    ax_bar.axvline(x=0, color='black', linewidth=0.5)
    ax_bar.invert_yaxis()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:2], orientation='horizontal',
                        fraction=0.05, pad=0.08, aspect=40)
    cbar.set_label('Patch contribution to similarity')

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    return fig


def visualize_component_spatial_distribution(
    det,
    pca_processor,
    gmm,
    component_idx: int,
    figsize: Tuple[int, int] = (6, 6),
    cmap: str = 'viridis',
) -> plt.Figure:
    """
    Visualize spatial distribution of a single GMM component on a detection.

    Shows which patches have high responsibility for a specific component,
    useful for understanding what each component has learned (e.g., head, stripes).

    Args:
        det: Detection object
        pca_processor: PCA processor
        gmm: GMM model
        component_idx: Which component to visualize
        figsize: Figure size
        cmap: Colormap

    Returns:
        matplotlib Figure
    """
    # Compute responsibilities
    resp = compute_patch_responsibilities(det.features, det.patch_mask, pca_processor, gmm)
    component_resp = resp[:, component_idx]

    # Load and crop image
    img = Image.open(det.image_path).convert('RGB')
    img_w, img_h = img.size
    x1, y1, x2, y2 = get_crop_bounds(det.square_crop_bbox, img_w, img_h)
    crop = img.crop((x1, y1, x2, y2))
    crop_w, crop_h = x2 - x1, y2 - y1

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(crop)

    coords = patch_coords_in_crop(det.patch_mask, crop_w, crop_h)
    cmap_obj = plt.get_cmap(cmap)
    norm = Normalize(vmin=0, vmax=component_resp.max())

    for idx, (px, py, pw, ph) in enumerate(coords):
        color = cmap_obj(norm(component_resp[idx]))
        ax.add_patch(Rectangle(
            (px, py), pw, ph,
            facecolor=color,
            edgecolor='none',
            alpha=0.7,
        ))

    ax.set_title(f'Component {component_idx} responsibility')
    ax.axis('off')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig
