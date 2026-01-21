"""Residual-based Fisher Vector visualization utilities.

Computes dual posteriors:
- Joint: p(k | s, r) where r = t - A*s - b
- Semantic: p(k | s) marginalized over residuals

The posterior shift (joint - semantic) is the key identity signal.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.special import logsumexp
import h5py


@dataclass
class ResidualFisherParams:
    """Parameters for residual-based Fisher encoding."""
    dim_s: int
    dim_t: int
    n_components: int
    A_global: np.ndarray  # [D_t, D_s] global regression
    b_global: np.ndarray  # [D_t]
    weights: np.ndarray   # [K]
    mu_s: np.ndarray      # [D_s, K]
    mu_r: np.ndarray      # [D_t, K] residual means
    vars_s: np.ndarray    # [D_s, K]
    vars_r: np.ndarray    # [D_t, K]


def load_residual_fisher_model(path: Path) -> ResidualFisherParams:
    """Load residual-based Fisher model from HDF5.

    Note: Julia stores matrices column-major, HDF5 reads into Python row-major.
    We transpose to get the expected shapes:
    - A_global: [D_t, D_s] for r = t - A*s - b
    - mu_s, vars_s: [D_s, K]
    - mu_r, vars_r: [D_t, K]
    """
    with h5py.File(path, "r") as f:
        return ResidualFisherParams(
            dim_s=int(f["dim_s"][()]),
            dim_t=int(f["dim_t"][()]),
            n_components=int(f["n_components"][()]),
            A_global=f["A_global"][:].T,  # [D_s, D_t] -> [D_t, D_s]
            b_global=f["b_global"][:],
            weights=f["weights"][:],
            mu_s=f["means_s"][:].T,  # [K, D_s] -> [D_s, K]
            mu_r=f["means_r"][:].T,  # [K, D_t] -> [D_t, K]
            vars_s=f["vars_s"][:].T,  # [K, D_s] -> [D_s, K]
            vars_r=f["vars_r"][:].T,  # [K, D_t] -> [D_t, K]
        )


def compute_residuals(S: np.ndarray, T: np.ndarray, params: ResidualFisherParams) -> np.ndarray:
    """Compute residuals r = t - A*s - b.

    Args:
        S: [N, D_s] semantic features
        T: [N, D_t] textural features
        params: model parameters

    Returns:
        R: [N, D_t] residuals
    """
    # R = T - S @ A.T - b
    return T - S @ params.A_global.T - params.b_global


def compute_joint_posteriors(
    S: np.ndarray,
    R: np.ndarray,
    params: ResidualFisherParams,
) -> np.ndarray:
    """Compute joint posteriors p(k | s, r).

    Args:
        S: [N, D_s] semantic features (PCA-transformed)
        R: [N, D_t] residuals
        params: model parameters

    Returns:
        posteriors: [N, K]
    """
    N = S.shape[0]
    K = params.n_components
    D_s = params.dim_s
    D_t = params.dim_t

    # Precompute log determinants
    log_det_s = np.sum(np.log(params.vars_s), axis=0)  # [K]
    log_det_r = np.sum(np.log(params.vars_r), axis=0)  # [K]

    inv_vars_s = 1.0 / params.vars_s  # [D_s, K]
    inv_vars_r = 1.0 / params.vars_r  # [D_t, K]

    log_2pi_s = D_s * np.log(2 * np.pi)
    log_2pi_r = D_t * np.log(2 * np.pi)

    log_weights = np.log(np.maximum(params.weights, 1e-10))

    log_prob = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        # Semantic log-likelihood
        diff_s = S - params.mu_s[:, k]  # [N, D_s]
        mahal_s = np.sum(diff_s**2 * inv_vars_s[:, k], axis=1)  # [N]
        log_p_s = -0.5 * (log_2pi_s + log_det_s[k] + mahal_s)

        # Residual log-likelihood
        diff_r = R - params.mu_r[:, k]  # [N, D_t]
        mahal_r = np.sum(diff_r**2 * inv_vars_r[:, k], axis=1)  # [N]
        log_p_r = -0.5 * (log_2pi_r + log_det_r[k] + mahal_r)

        log_prob[:, k] = log_weights[k] + log_p_s + log_p_r

    log_sum = logsumexp(log_prob, axis=1, keepdims=True)
    return np.exp(log_prob - log_sum).astype(np.float32)


def compute_semantic_posteriors(
    S: np.ndarray,
    params: ResidualFisherParams,
) -> np.ndarray:
    """Compute semantic-only posteriors p(k | s).

    Args:
        S: [N, D_s] semantic features (PCA-transformed)
        params: model parameters

    Returns:
        posteriors: [N, K]
    """
    N = S.shape[0]
    K = params.n_components
    D_s = params.dim_s

    log_det_s = np.sum(np.log(params.vars_s), axis=0)  # [K]
    inv_vars_s = 1.0 / params.vars_s  # [D_s, K]
    log_2pi_s = D_s * np.log(2 * np.pi)
    log_weights = np.log(np.maximum(params.weights, 1e-10))

    log_prob = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        diff_s = S - params.mu_s[:, k]
        mahal_s = np.sum(diff_s**2 * inv_vars_s[:, k], axis=1)
        log_prob[:, k] = log_weights[k] - 0.5 * (log_2pi_s + log_det_s[k] + mahal_s)

    log_sum = logsumexp(log_prob, axis=1, keepdims=True)
    return np.exp(log_prob - log_sum).astype(np.float32)


def extract_valid_features(
    semantic_features,
    textural_features,
    patch_mask,
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
    mask_flat = patch_mask.flatten()

    # Convert to numpy if needed
    if hasattr(sem_flat, 'numpy'):
        sem_flat = sem_flat.numpy()
    if hasattr(tex_flat, 'numpy'):
        tex_flat = tex_flat.numpy()
    if hasattr(mask_flat, 'numpy'):
        mask_flat = mask_flat.numpy()

    mask_flat = mask_flat.astype(bool)

    S = sem_flat[mask_flat]
    T = tex_flat[mask_flat]

    if semantic_pca is not None:
        S = semantic_pca.transform(S).astype(np.float32)
    if textural_pca is not None:
        T = textural_pca.transform(T).astype(np.float32)

    return S, T


def compute_residual_fisher_joint_responsibilities(
    semantic_features,
    textural_features,
    patch_mask,
    params: ResidualFisherParams,
    semantic_pca=None,
    textural_pca=None,
) -> np.ndarray:
    """Compute joint responsibilities p(k|s,r) for valid patches.

    Args:
        semantic_features: [D_s, H, W] semantic features
        textural_features: [D_t, H, W] textural features
        patch_mask: [H, W] boolean mask
        params: Residual Fisher model parameters
        semantic_pca: Optional PCA for semantic features
        textural_pca: Optional PCA for textural features

    Returns:
        responsibilities: [N_valid, K] joint posteriors
    """
    S, T = extract_valid_features(
        semantic_features, textural_features, patch_mask,
        semantic_pca, textural_pca
    )

    if len(S) == 0:
        return np.empty((0, params.n_components))

    # Compute residuals
    R = compute_residuals(S, T, params)

    return compute_joint_posteriors(S, R, params)


def compute_residual_fisher_semantic_responsibilities(
    semantic_features,
    patch_mask,
    params: ResidualFisherParams,
    semantic_pca=None,
) -> np.ndarray:
    """Compute semantic-only responsibilities p(k|s) for valid patches.

    Args:
        semantic_features: [D_s, H, W] semantic features
        patch_mask: [H, W] boolean mask
        params: Residual Fisher model parameters
        semantic_pca: Optional PCA for semantic features

    Returns:
        responsibilities: [N_valid, K] semantic posteriors
    """
    d_s = semantic_features.shape[0]

    sem_flat = semantic_features.reshape(d_s, -1).T
    mask_flat = patch_mask.flatten()

    if hasattr(sem_flat, 'numpy'):
        sem_flat = sem_flat.numpy()
    if hasattr(mask_flat, 'numpy'):
        mask_flat = mask_flat.numpy()

    mask_flat = mask_flat.astype(bool)

    S = sem_flat[mask_flat]

    if len(S) == 0:
        return np.empty((0, params.n_components))

    if semantic_pca is not None:
        S = semantic_pca.transform(S).astype(np.float32)

    return compute_semantic_posteriors(S, params)


def compute_posterior_shift(
    semantic_features,
    textural_features,
    patch_mask,
    params: ResidualFisherParams,
    semantic_pca=None,
    textural_pca=None,
) -> np.ndarray:
    """Compute posterior shift (ρ - γ) for valid patches.

    The posterior shift captures how much the residual information
    changes the component assignment - the key identity signal.

    Args:
        semantic_features: [D_s, H, W] semantic features
        textural_features: [D_t, H, W] textural features
        patch_mask: [H, W] boolean mask
        params: Residual Fisher model parameters
        semantic_pca: Optional PCA for semantic features
        textural_pca: Optional PCA for textural features

    Returns:
        posterior_shift: [N_valid, K] (joint - semantic posteriors)
    """
    joint = compute_residual_fisher_joint_responsibilities(
        semantic_features, textural_features, patch_mask,
        params, semantic_pca, textural_pca
    )
    semantic = compute_residual_fisher_semantic_responsibilities(
        semantic_features, patch_mask, params, semantic_pca
    )

    return joint - semantic
