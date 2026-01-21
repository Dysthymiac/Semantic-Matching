"""Joint Fisher Vector encoding using Mixture of Linear Regressions.

Encodes p(t|s,c) - how textural features deviate within each semantic region.
For individual re-identification: captures texture patterns (e.g., zebra stripes)
within corresponding body regions across different images.

Fisher gradients are computed w.r.t. textural parameters (μ_t, σ²_t) for each
component, weighted by posterior p(c|s,t).

Output dimension: K × 2 × D_t (K components, gradients for μ_t and σ²_t)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import logsumexp

from ..codebook.mixture_of_linear_regressions import MLRParameters


def compute_semantic_posteriors(
    S: np.ndarray,
    params: MLRParameters,
) -> np.ndarray:
    """Compute marginalized semantic posteriors p(c | s).

    Ignores textural features - uses only the semantic Gaussian.

    Args:
        S: Semantic features [N, D_s]
        params: MLR model parameters

    Returns:
        posteriors: [N, C] posterior probabilities
    """
    N = S.shape[0]
    C = params.n_components

    log_norm_s = -0.5 * np.sum(np.log(2 * np.pi * params.semantic_vars), axis=1)
    inv_sem_vars = 1.0 / params.semantic_vars

    log_prob = np.zeros((N, C), dtype=np.float64)
    for c in range(C):
        diff_s = S - params.semantic_means[c]
        log_prob[:, c] = -0.5 * np.sum(diff_s**2 * inv_sem_vars[c], axis=1) + log_norm_s[c]

    log_weights = np.log(np.maximum(params.weights, 1e-10))
    log_prob_weighted = log_prob + log_weights

    log_sum = logsumexp(log_prob_weighted, axis=1, keepdims=True)
    return np.exp(log_prob_weighted - log_sum).astype(np.float32)


def compute_mlr_log_likelihood(
    S: np.ndarray,
    T: np.ndarray,
    params: MLRParameters,
) -> np.ndarray:
    """Compute log p(s, t | c) for each sample and component.

    Args:
        S: Semantic features [N, D_s]
        T: Textural features [N, D_t]
        params: MLR model parameters

    Returns:
        Log-likelihoods [N, C]
    """
    N = S.shape[0]
    C = params.n_components

    # Pre-compute log normalization constants
    log_norm_s = -0.5 * np.sum(np.log(2 * np.pi * params.semantic_vars), axis=1)
    log_norm_t = -0.5 * np.sum(np.log(2 * np.pi * params.residual_vars), axis=1)

    # Inverse variances for speed
    inv_sem_vars = 1.0 / params.semantic_vars
    inv_res_vars = 1.0 / params.residual_vars

    log_prob = np.zeros((N, C), dtype=np.float64)

    for c in range(C):
        # p(s|c): semantic marginal
        diff_s = S - params.semantic_means[c]
        log_p_s = -0.5 * np.sum(diff_s**2 * inv_sem_vars[c], axis=1) + log_norm_s[c]

        # p(t|s,c): textural conditional with regression
        cond_mean = params.textural_means[c] + diff_s @ params.regression_A[c].T
        diff_t = T - cond_mean
        log_p_t = -0.5 * np.sum(diff_t**2 * inv_res_vars[c], axis=1) + log_norm_t[c]

        log_prob[:, c] = log_p_s + log_p_t

    return log_prob


def compute_mlr_posteriors(
    S: np.ndarray,
    T: np.ndarray,
    params: MLRParameters,
) -> np.ndarray:
    """Compute posterior probabilities p(c | s, t) using MLR model.

    Args:
        S: Semantic features [N, D_s]
        T: Textural features [N, D_t]
        params: MLR model parameters

    Returns:
        posteriors: [N, C] posterior probabilities
    """
    log_prob = compute_mlr_log_likelihood(S, T, params)

    # Mask out zero-weight components (empty clusters)
    log_weights = np.log(np.maximum(params.weights, 1e-10))
    log_prob_weighted = log_prob + log_weights

    log_sum = logsumexp(log_prob_weighted, axis=1, keepdims=True)
    return np.exp(log_prob_weighted - log_sum).astype(np.float32)


def compute_textural_residuals(
    S: np.ndarray,
    T: np.ndarray,
    params: MLRParameters,
) -> np.ndarray:
    """Compute textural residuals r = t - μ_t - A(s - μ_s) for all components.

    Args:
        S: Semantic features [N, D_s]
        T: Textural features [N, D_t]
        params: MLR model parameters

    Returns:
        residuals: [N, C, D_t] residuals for each sample and component
    """
    N = S.shape[0]
    C = params.n_components
    D_t = params.n_textural

    residuals = np.zeros((N, C, D_t), dtype=np.float32)

    for c in range(C):
        diff_s = S - params.semantic_means[c]
        predicted = params.textural_means[c] + diff_s @ params.regression_A[c].T
        residuals[:, c, :] = T - predicted

    return residuals


def encode_mlr_fisher_vector(
    S: np.ndarray,
    T: np.ndarray,
    params: MLRParameters,
    power_norm_alpha: float = 0.5,
) -> np.ndarray:
    """Encode features as Fisher Vector using MLR model.

    Computes Fisher gradients w.r.t. textural parameters (μ_t, σ²_t) for each
    component, weighted by posteriors p(c|s,t).

    Args:
        S: Semantic features [N, D_s]
        T: Textural features [N, D_t]
        params: MLR model parameters
        power_norm_alpha: Exponent for power normalization (default 0.5)

    Returns:
        Fisher Vector [2 * K * D_t] with power + L2 normalization
    """
    n_patches = S.shape[0]
    C = params.n_components
    D_t = params.n_textural

    if n_patches == 0:
        return np.zeros(2 * C * D_t, dtype=np.float32)

    # Compute posteriors and residuals
    gamma = compute_mlr_posteriors(S, T, params)  # [N, C]
    residuals = compute_textural_residuals(S, T, params)  # [N, C, D_t]

    # Normalization factors from MLR weights
    sqrt_w = np.sqrt(params.weights + 1e-10)
    sqrt_2w = np.sqrt(2 * params.weights + 1e-10)

    # Fisher gradients
    gradients_mu = np.zeros((C, D_t), dtype=np.float32)
    gradients_sigma = np.zeros((C, D_t), dtype=np.float32)

    for c in range(C):
        # Normalized residual: r / σ_t
        r_norm = residuals[:, c, :] / np.sqrt(params.residual_vars[c] + 1e-10)

        # Gradient w.r.t. μ_t: sum_i γ_ic * (r / σ_t) / sqrt(w_c)
        gradients_mu[c] = (gamma[:, c:c+1] * r_norm).sum(axis=0) / sqrt_w[c]

        # Gradient w.r.t. σ²_t: sum_i γ_ic * (r²/σ² - 1) / sqrt(2w_c)
        r_sq_norm = r_norm ** 2 - 1
        gradients_sigma[c] = (gamma[:, c:c+1] * r_sq_norm).sum(axis=0) / sqrt_2w[c]

    # Concatenate: [2 * K * D_t]
    fv = np.concatenate([gradients_mu.flatten(), gradients_sigma.flatten()])

    # Power normalization: sign(x) * |x|^alpha
    fv = np.sign(fv) * np.abs(fv) ** power_norm_alpha

    # L2 normalization
    norm = np.linalg.norm(fv)
    if norm > 1e-10:
        fv = fv / norm

    return fv.astype(np.float32)


def encode_detection_mlr_fisher_vector(
    semantic_features: np.ndarray,
    textural_features: np.ndarray,
    patch_mask: np.ndarray,
    params: MLRParameters,
    semantic_pca=None,
    textural_pca=None,
    power_norm_alpha: float = 0.5,
) -> Optional[np.ndarray]:
    """Encode a detection with spatially aligned features as MLR Fisher Vector.

    Args:
        semantic_features: [D_s, H, W] semantic features
        textural_features: [D_t, H, W] textural features
        patch_mask: [H, W] boolean mask of valid patches
        params: MLR model parameters
        semantic_pca: Optional PCA for semantic features
        textural_pca: Optional PCA for textural features
        power_norm_alpha: Exponent for power normalization

    Returns:
        Fisher Vector or None if no valid patches
    """
    # Flatten spatial dimensions
    d_s, h, w = semantic_features.shape
    d_t = textural_features.shape[0]

    semantic_flat = semantic_features.reshape(d_s, -1).T  # [H*W, D_s]
    textural_flat = textural_features.reshape(d_t, -1).T  # [H*W, D_t]
    mask_flat = patch_mask.flatten().astype(bool)

    # Select valid patches
    valid_semantic = semantic_flat[mask_flat]
    valid_textural = textural_flat[mask_flat]

    if valid_semantic.shape[0] == 0:
        return None

    # Apply PCA if provided
    if semantic_pca is not None:
        valid_semantic = semantic_pca.transform(valid_semantic).astype(np.float32)
    if textural_pca is not None:
        valid_textural = textural_pca.transform(valid_textural).astype(np.float32)

    return encode_mlr_fisher_vector(
        valid_semantic,
        valid_textural,
        params,
        power_norm_alpha,
    )


def get_mlr_fv_dimension(params: MLRParameters) -> int:
    """Compute output dimension of MLR Fisher Vector.

    Args:
        params: MLR model parameters

    Returns:
        Output dimension: 2 * K * D_t
    """
    return 2 * params.n_components * params.n_textural
