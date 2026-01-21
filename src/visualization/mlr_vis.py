"""MLR (Mixture of Linear Regressions) visualization utilities."""

import numpy as np
import torch

from ..codebook.mixture_of_linear_regressions import MLRParameters
from ..features.joint_fisher_vector import compute_mlr_posteriors, compute_semantic_posteriors


def extract_valid_features(
    semantic_features: torch.Tensor,
    textural_features: torch.Tensor,
    patch_mask: torch.Tensor,
    semantic_pca=None,
    textural_pca=None,
):
    """Extract valid patch features and optionally apply PCA.

    Args:
        semantic_features: [D_s, H, W] semantic features
        textural_features: [D_t, H, W] textural features
        patch_mask: [H, W] boolean mask
        semantic_pca: Optional PCA for semantic features
        textural_pca: Optional PCA for textural features

    Returns:
        S: [N_valid, D_s'] semantic features (PCA-transformed if provided)
        T: [N_valid, D_t'] textural features (PCA-transformed if provided)
    """
    d_s, h, w = semantic_features.shape
    d_t = textural_features.shape[0]

    sem_flat = semantic_features.reshape(d_s, -1).T  # [H*W, D_s]
    tex_flat = textural_features.reshape(d_t, -1).T  # [H*W, D_t]
    mask_flat = patch_mask.flatten().bool()

    if isinstance(sem_flat, torch.Tensor):
        sem_flat = sem_flat.numpy()
    if isinstance(tex_flat, torch.Tensor):
        tex_flat = tex_flat.numpy()
    if isinstance(mask_flat, torch.Tensor):
        mask_flat = mask_flat.numpy()

    S = sem_flat[mask_flat]
    T = tex_flat[mask_flat]

    if semantic_pca is not None:
        S = semantic_pca.transform(S).astype(np.float32)
    if textural_pca is not None:
        T = textural_pca.transform(T).astype(np.float32)

    return S, T


def compute_mlr_patch_responsibilities(
    semantic_features: torch.Tensor,
    textural_features: torch.Tensor,
    patch_mask: torch.Tensor,
    params: MLRParameters,
    semantic_pca=None,
    textural_pca=None,
) -> np.ndarray:
    """Compute joint MLR responsibilities p(c|s,t) for valid patches.

    Args:
        semantic_features: [D_s, H, W] semantic features
        textural_features: [D_t, H, W] textural features
        patch_mask: [H, W] boolean mask
        params: MLR model parameters
        semantic_pca: Optional PCA for semantic features
        textural_pca: Optional PCA for textural features

    Returns:
        responsibilities: [N_valid, C] joint posteriors
    """
    S, T = extract_valid_features(
        semantic_features, textural_features, patch_mask,
        semantic_pca, textural_pca
    )

    if len(S) == 0:
        return np.empty((0, params.n_components))

    return compute_mlr_posteriors(S, T, params)


def compute_semantic_patch_responsibilities(
    semantic_features: torch.Tensor,
    patch_mask: torch.Tensor,
    params: MLRParameters,
    semantic_pca=None,
) -> np.ndarray:
    """Compute marginalized semantic responsibilities p(c|s) for valid patches.

    Args:
        semantic_features: [D_s, H, W] semantic features
        patch_mask: [H, W] boolean mask
        params: MLR model parameters
        semantic_pca: Optional PCA for semantic features

    Returns:
        responsibilities: [N_valid, C] semantic posteriors
    """
    d_s = semantic_features.shape[0]

    sem_flat = semantic_features.reshape(d_s, -1).T
    mask_flat = patch_mask.flatten().bool()

    if isinstance(sem_flat, torch.Tensor):
        sem_flat = sem_flat.numpy()
    if isinstance(mask_flat, torch.Tensor):
        mask_flat = mask_flat.numpy()

    S = sem_flat[mask_flat]

    if len(S) == 0:
        return np.empty((0, params.n_components))

    if semantic_pca is not None:
        S = semantic_pca.transform(S).astype(np.float32)

    return compute_semantic_posteriors(S, params)
