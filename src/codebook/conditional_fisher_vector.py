"""Conditional Fisher Vector for re-identification.

Two-stage model:
1. Semantic GMM fitted on semantic features only - components are pure pose clusters
2. Per-component regression fitted with semantic posteriors as weights

This ensures pose determines correspondence, then texture deviations encode identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


@dataclass
class ConditionalFisherModel:
    """Parameters for Conditional Fisher Vector."""

    n_components: int
    dim_s: int
    dim_t: int

    # Semantic GMM parameters (from pre-fitted GMM on semantic features)
    weights: np.ndarray       # [K]
    means_s: np.ndarray       # [K, D_s]
    vars_s: np.ndarray        # [K, D_s] diagonal

    # Per-component regression: t = A_k @ s + b_k
    A: np.ndarray             # [K, D_t, D_s]
    b: np.ndarray             # [K, D_t]

    # Conditional variance
    vars_t_given_s: np.ndarray  # [K, D_t]

    def save(self, path: Path) -> None:
        """Save to HDF5."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w') as f:
            f.create_dataset('n_components', data=self.n_components)
            f.create_dataset('dim_s', data=self.dim_s)
            f.create_dataset('dim_t', data=self.dim_t)
            f.create_dataset('weights', data=self.weights)
            f.create_dataset('means_s', data=self.means_s)
            f.create_dataset('vars_s', data=self.vars_s)
            f.create_dataset('A', data=self.A)
            f.create_dataset('b', data=self.b)
            f.create_dataset('vars_t_given_s', data=self.vars_t_given_s)

    @classmethod
    def load(cls, path: Path) -> 'ConditionalFisherModel':
        """Load from HDF5."""
        with h5py.File(path, 'r') as f:
            return cls(
                n_components=int(f['n_components'][()]),
                dim_s=int(f['dim_s'][()]),
                dim_t=int(f['dim_t'][()]),
                weights=f['weights'][:],
                means_s=f['means_s'][:],
                vars_s=f['vars_s'][:],
                A=f['A'][:],
                b=f['b'][:],
                vars_t_given_s=f['vars_t_given_s'][:],
            )


def compute_semantic_posteriors_from_gmm(
    S: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    vars: np.ndarray,
) -> np.ndarray:
    """Compute γ_ik = p(k|s_i) from semantic GMM.

    Args:
        S: [N, D_s] semantic features
        weights: [K] mixture weights
        means: [K, D_s] component means
        vars: [K, D_s] component variances (diagonal)

    Returns:
        gamma: [N, K] posterior probabilities
    """
    N = S.shape[0]
    K = len(weights)

    log_weights = np.log(weights + 1e-10)
    log_det = np.sum(np.log(vars), axis=1)  # [K]
    inv_vars = 1.0 / vars  # [K, D_s]

    log_prob = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        diff = S - means[k]
        mahal = np.sum(diff**2 * inv_vars[k], axis=1)
        log_prob[:, k] = log_weights[k] - 0.5 * (log_det[k] + mahal)

    log_sum = logsumexp(log_prob, axis=1, keepdims=True)
    return np.exp(log_prob - log_sum).astype(np.float32)


def fit_per_component_regression(
    S: np.ndarray,
    T: np.ndarray,
    gamma: np.ndarray,
    k: int,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit weighted least squares regression for component k.

    Args:
        S: [N, D_s] semantic features
        T: [N, D_t] textural features
        gamma: [N, K] semantic posteriors
        k: component index
        regularization: ridge regularization

    Returns:
        A_k: [D_t, D_s] regression matrix
        b_k: [D_t] bias
        var_k: [D_t] conditional variance
    """
    w = gamma[:, k].astype(np.float64)
    w_sum = w.sum()

    D_s = S.shape[1]
    D_t = T.shape[1]

    if w_sum < 1e-10:
        return (
            np.zeros((D_t, D_s), dtype=np.float32),
            np.zeros(D_t, dtype=np.float32),
            np.ones(D_t, dtype=np.float32),
        )

    # Weighted means
    s_bar = np.average(S, weights=w, axis=0)
    t_bar = np.average(T, weights=w, axis=0)

    # Centered data
    S_centered = S - s_bar
    T_centered = T - t_bar

    # Weighted covariance of S: [D_s, D_s]
    Css = (S_centered.T * w) @ S_centered / w_sum
    Css += regularization * np.eye(D_s)

    # Weighted cross-covariance: [D_t, D_s]
    Cts = (T_centered.T * w) @ S_centered / w_sum

    # A_k = Cts @ Css^{-1}
    A_k = np.linalg.solve(Css.T, Cts.T).T

    # b_k = t̄ - A_k @ s̄
    b_k = t_bar - A_k @ s_bar

    # Residuals and conditional variance
    residuals = T - (S @ A_k.T + b_k)
    var_k = np.average(residuals**2, weights=w, axis=0)
    var_k = np.maximum(var_k, 1e-10)

    return A_k.astype(np.float32), b_k.astype(np.float32), var_k.astype(np.float32)


def fit_conditional_model(
    S: np.ndarray,
    T: np.ndarray,
    gmm: GaussianMixture,
    regularization: float = 1e-6,
) -> ConditionalFisherModel:
    """Fit per-component regressions given pre-trained semantic GMM.

    Args:
        S: [N, D_s] semantic features (PCA-transformed)
        T: [N, D_t] textural features (PCA-transformed)
        gmm: pre-fitted GaussianMixture on semantic features
        regularization: ridge regularization for least squares

    Returns:
        Fitted ConditionalFisherModel
    """
    N, D_s = S.shape
    D_t = T.shape[1]
    K = gmm.n_components

    print(f"Fitting conditional model:")
    print(f"  N = {N:,} samples")
    print(f"  D_s = {D_s}, D_t = {D_t}, K = {K}")

    # Extract GMM parameters
    weights = gmm.weights_.astype(np.float32)
    means_s = gmm.means_.astype(np.float32)

    if gmm.covariance_type == 'diag':
        vars_s = gmm.covariances_.astype(np.float32)
    elif gmm.covariance_type == 'full':
        vars_s = np.array([np.diag(cov) for cov in gmm.covariances_]).astype(np.float32)
    elif gmm.covariance_type == 'spherical':
        vars_s = np.tile(gmm.covariances_[:, np.newaxis], (1, D_s)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported covariance type: {gmm.covariance_type}")

    # Compute semantic posteriors
    print("Computing semantic posteriors...")
    gamma = compute_semantic_posteriors_from_gmm(S, weights, means_s, vars_s)

    # Fit per-component regressions
    print("Fitting per-component regressions...")
    A = np.zeros((K, D_t, D_s), dtype=np.float32)
    b = np.zeros((K, D_t), dtype=np.float32)
    vars_t_given_s = np.zeros((K, D_t), dtype=np.float32)

    for k in tqdm(range(K), desc="Components"):
        A[k], b[k], vars_t_given_s[k] = fit_per_component_regression(
            S, T, gamma, k, regularization
        )

    return ConditionalFisherModel(
        n_components=K,
        dim_s=D_s,
        dim_t=D_t,
        weights=weights,
        means_s=means_s,
        vars_s=vars_s,
        A=A,
        b=b,
        vars_t_given_s=vars_t_given_s,
    )


def compute_semantic_posteriors(
    S: np.ndarray,
    model: ConditionalFisherModel,
) -> np.ndarray:
    """Compute γ_ik = p(k|s_i) using model parameters."""
    return compute_semantic_posteriors_from_gmm(
        S, model.weights, model.means_s, model.vars_s
    )


def compute_residuals(
    S: np.ndarray,
    T: np.ndarray,
    model: ConditionalFisherModel,
) -> np.ndarray:
    """Compute per-component residuals r̃_ik = t_i - A_k @ s_i - b_k.

    Args:
        S: [N, D_s] semantic features
        T: [N, D_t] textural features
        model: fitted model

    Returns:
        residuals: [N, K, D_t]
    """
    N = S.shape[0]
    K = model.n_components
    D_t = model.dim_t

    residuals = np.zeros((N, K, D_t), dtype=np.float32)
    for k in range(K):
        predicted = S @ model.A[k].T + model.b[k]
        residuals[:, k, :] = T - predicted

    return residuals


def encode_fisher_vector(
    S: np.ndarray,
    T: np.ndarray,
    model: ConditionalFisherModel,
    power_norm_alpha: float = 0.5,
) -> np.ndarray:
    """Encode as Conditional Fisher Vector.

    Uses semantic-only routing (γ_ik) not joint posterior.

    Args:
        S: [N, D_s] semantic features
        T: [N, D_t] textural features
        model: fitted model
        power_norm_alpha: power normalization exponent

    Returns:
        fv: [2 * K * D_t] Fisher vector
    """
    K = model.n_components
    D_t = model.dim_t

    if S.shape[0] == 0:
        return np.zeros(2 * K * D_t, dtype=np.float32)

    # Semantic posteriors for routing
    gamma = compute_semantic_posteriors(S, model)

    # Per-component residuals
    residuals = compute_residuals(S, T, model)

    # Normalized residuals
    sigma = np.sqrt(model.vars_t_given_s)  # [K, D_t]
    norm_residuals = residuals / sigma[np.newaxis, :, :]  # [N, K, D_t]

    # Fisher gradients
    sqrt_w = np.sqrt(model.weights + 1e-10)

    # Mean gradient: (1/√w_k) Σ_i γ_ik (r̃_ik / σ_k)
    G_mu = np.einsum('nk,nkd->kd', gamma, norm_residuals) / sqrt_w[:, np.newaxis]

    # Variance gradient: (1/√(2w_k)) Σ_i γ_ik ((r̃_ik/σ_k)² - 1)
    G_sigma = np.einsum('nk,nkd->kd', gamma, norm_residuals**2 - 1) / (np.sqrt(2) * sqrt_w[:, np.newaxis])

    # Concatenate
    fv = np.concatenate([G_mu.flatten(), G_sigma.flatten()])

    # Power normalization
    fv = np.sign(fv) * np.abs(fv) ** power_norm_alpha

    # L2 normalization
    norm = np.linalg.norm(fv)
    if norm > 1e-10:
        fv = fv / norm

    return fv.astype(np.float32)


def encode_detection(
    semantic_features: np.ndarray,
    textural_features: np.ndarray,
    patch_mask: np.ndarray,
    model: ConditionalFisherModel,
    semantic_pca=None,
    textural_pca=None,
    power_norm_alpha: float = 0.5,
) -> np.ndarray | None:
    """Encode a detection as Conditional Fisher Vector.

    Args:
        semantic_features: [D_s, H, W]
        textural_features: [D_t, H, W]
        patch_mask: [H, W]
        model: fitted model
        semantic_pca: optional PCA for semantic features
        textural_pca: optional PCA for textural features
        power_norm_alpha: power normalization exponent

    Returns:
        Fisher vector or None if no valid patches
    """
    d_s, h, w = semantic_features.shape
    d_t = textural_features.shape[0]

    sem_flat = semantic_features.reshape(d_s, -1).T
    tex_flat = textural_features.reshape(d_t, -1).T
    mask_flat = patch_mask.flatten().astype(bool)

    S = sem_flat[mask_flat]
    T = tex_flat[mask_flat]

    if S.shape[0] == 0:
        return None

    if semantic_pca is not None:
        S = semantic_pca.transform(S).astype(np.float32)
    if textural_pca is not None:
        T = textural_pca.transform(T).astype(np.float32)

    return encode_fisher_vector(S, T, model, power_norm_alpha)


def get_fv_dimension(model: ConditionalFisherModel) -> int:
    """Get Fisher vector output dimension."""
    return 2 * model.n_components * model.dim_t
