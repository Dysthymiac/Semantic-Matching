"""Mixture of Linear Regressions model for joint semantic-textural modeling.

This model captures p(t|s) as a mixture where:
- Each component has a linear regression from semantic to textural
- Diagonal covariances within each modality
- Full cross-covariance captured by regression matrix A

Model per component c:
    p(s | c) = N(s | μₛᶜ, Dₛᶜ)           -- semantic marginal (diagonal cov)
    p(t | s, c) = N(t | μₜᶜ + Aᶜ(s - μₛᶜ), Dₜ|ₛᶜ)  -- textural conditional (diagonal cov)

This is equivalent to a joint GMM with structured covariance:
    Σᶜ = [Dₛᶜ      Dₛᶜ Aᶜᵀ           ]
         [Aᶜ Dₛᶜ   Aᶜ Dₛᶜ Aᶜᵀ + Dₜ|ₛᶜ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import pickle
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


@dataclass
class MLRParameters:
    """Parameters for Mixture of Linear Regressions model."""
    n_components: int
    n_semantic: int
    n_textural: int

    # Mixing weights [C]
    weights: np.ndarray

    # Semantic parameters
    semantic_means: np.ndarray    # [C, D_s]
    semantic_vars: np.ndarray     # [C, D_s] (diagonal)

    # Textural/regression parameters
    textural_means: np.ndarray    # [C, D_t] (intercept)
    regression_A: np.ndarray      # [C, D_t, D_s]
    residual_vars: np.ndarray     # [C, D_t] (diagonal)

    def save(self, path: Path):
        """Save parameters to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> 'MLRParameters':
        """Load parameters from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_mlr_from_npz(path: Path) -> MLRParameters:
    """Load MLR parameters from Julia-generated NPZ file.

    Julia saves arrays in [C, D] format for vectors and [C, D_t, D_s] for matrices.

    Args:
        path: Path to .npz file from Julia fit_mlr.jl

    Returns:
        MLRParameters with loaded model
    """
    data = np.load(path)

    n_components = int(data["n_components"])
    D_s = int(data["D_s"])
    D_t = int(data["D_t"])

    return MLRParameters(
        n_components=n_components,
        n_semantic=D_s,
        n_textural=D_t,
        weights=data["weights"].astype(np.float32),
        semantic_means=data["mu_s"].astype(np.float32),
        semantic_vars=data["sigma2_s"].astype(np.float32),
        textural_means=data["mu_t"].astype(np.float32),
        regression_A=data["A"].astype(np.float32),
        residual_vars=data["sigma2_t"].astype(np.float32),
    )


def load_structured_gmm(path: Path) -> MLRParameters:
    """Load Structured GMM from Julia-generated HDF5 and convert to MLRParameters.

    The Structured GMM has cross-covariance which is converted to:
    - regression_A = cross_cov @ diag(1/vars_s)
    - residual_vars = diag(Σ_{t|s}) = vars_t - diag(A @ cross_cov.T)

    Args:
        path: Path to .h5 file from Julia fit_structured_gmm.jl

    Returns:
        MLRParameters with computed derived values
    """
    import h5py

    with h5py.File(path, 'r') as f:
        K = int(f["n_components"][()])
        D_s = int(f["dim_s"][()])
        D_t = int(f["dim_t"][()])

        weights = f["weights"][:].astype(np.float32)
        means_s = f["means_s"][:].T.astype(np.float32)  # [D_s, K] -> [K, D_s]
        means_t = f["means_t"][:].T.astype(np.float32)  # [D_t, K] -> [K, D_t]
        vars_s = f["vars_s"][:].T.astype(np.float32)    # [D_s, K] -> [K, D_s]
        vars_t = f["vars_t"][:].T.astype(np.float32)    # [D_t, K] -> [K, D_t]
        cross_cov = np.transpose(f["cross_cov"][:], (2, 0, 1)).astype(np.float32)  # [D_t, D_s, K] -> [K, D_t, D_s]

    inv_vars_s = 1.0 / vars_s  # [K, D_s]

    # Regression matrix: A = Σ_ts @ diag(1/σ²_s)
    regression_A = cross_cov * inv_vars_s[:, None, :]  # [K, D_t, D_s]

    # Conditional variance: diag(Σ_{t|s}) = σ²_t - diag(A @ Σ_st')
    residual_vars = np.zeros((K, D_t), dtype=np.float32)
    for k in range(K):
        correction = np.sum(regression_A[k] * cross_cov[k], axis=1)
        residual_vars[k] = np.maximum(vars_t[k] - correction, 1e-6)

    return MLRParameters(
        n_components=K,
        n_semantic=D_s,
        n_textural=D_t,
        weights=weights,
        semantic_means=means_s,
        semantic_vars=vars_s,
        textural_means=means_t,
        regression_A=regression_A,
        residual_vars=residual_vars,
    )


class MixtureOfLinearRegressions:
    """Mixture of Linear Regressions fitted via EM algorithm."""

    def __init__(
        self,
        n_components: int = 64,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-3,  # Increased for numerical stability
        random_state: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize the model.

        Args:
            n_components: Number of mixture components
            max_iter: Maximum EM iterations
            tol: Convergence tolerance (relative change in log-likelihood)
            reg_covar: Regularization added to variances for numerical stability
            random_state: Random seed for reproducibility
            verbose: Print progress during fitting
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.verbose = verbose

        # Will be set after fitting
        self.params: Optional[MLRParameters] = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.log_likelihood_history_: list = []

    def _initialize_parameters(self, S: np.ndarray, T: np.ndarray):
        """Initialize parameters using GMM on concatenated features.

        Fits a diagonal GMM on [s, t] concatenated features, then uses
        the GMM's weights, means, and variances to initialize the MLR.
        Regression matrices A are initialized to zero.

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]
        """
        N, D_s = S.shape
        D_t = T.shape[1]
        C = self.n_components

        if self.verbose:
            print(f"Initializing {C} components with GaussianMixture (diag)...")

        # Fit GMM on concatenated features
        ST = np.hstack([S, T])
        gmm = GaussianMixture(
            n_components=C,
            covariance_type='diag',
            random_state=self.random_state,
            n_init=1,
            max_iter=100,
            verbose=2 if self.verbose else 0,
            verbose_interval=1,
        )
        gmm.fit(ST)

        if self.verbose:
            print(f"GMM converged: {gmm.converged_}, iterations: {gmm.n_iter_}")

        # Extract parameters from GMM
        # means_: [C, D_s + D_t], covariances_: [C, D_s + D_t] for diag
        weights = gmm.weights_.astype(np.float32)
        semantic_means = gmm.means_[:, :D_s].astype(np.float32)
        textural_means = gmm.means_[:, D_s:].astype(np.float32)
        semantic_vars = (gmm.covariances_[:, :D_s] + self.reg_covar).astype(np.float32)
        residual_vars = (gmm.covariances_[:, D_s:] + self.reg_covar).astype(np.float32)

        # Initialize regression matrices to zero
        regression_A = np.zeros((C, D_t, D_s), dtype=np.float32)

        self.params = MLRParameters(
            n_components=C,
            n_semantic=D_s,
            n_textural=D_t,
            weights=weights,
            semantic_means=semantic_means,
            semantic_vars=semantic_vars,
            textural_means=textural_means,
            regression_A=regression_A,
            residual_vars=residual_vars,
        )

        if self.verbose:
            print(f"Initialization complete. Component sizes: min={weights.min()*N:.0f}, max={weights.max()*N:.0f}")

    def _compute_log_likelihood_per_component(
        self, S: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """Compute log p(s, t | c) for each sample and component.

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]

        Returns:
            Log-likelihoods [N, C]
        """
        N = S.shape[0]
        C = self.n_components
        log_prob = np.zeros((N, C))

        for c in range(C):
            # Log p(s | c) - semantic marginal
            diff_s = S - self.params.semantic_means[c]  # [N, D_s]
            log_p_s = -0.5 * np.sum(
                diff_s**2 / self.params.semantic_vars[c] +
                np.log(2 * np.pi * self.params.semantic_vars[c]),
                axis=1
            )

            # Conditional mean: μₜ + A(s - μₛ)
            cond_mean = (
                self.params.textural_means[c] +
                diff_s @ self.params.regression_A[c].T
            )  # [N, D_t]

            # Log p(t | s, c) - textural conditional
            diff_t = T - cond_mean
            log_p_t_given_s = -0.5 * np.sum(
                diff_t**2 / self.params.residual_vars[c] +
                np.log(2 * np.pi * self.params.residual_vars[c]),
                axis=1
            )

            log_prob[:, c] = log_p_s + log_p_t_given_s

        return log_prob

    def _e_step(self, S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, float]:
        """E-step: compute responsibilities.

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]

        Returns:
            responsibilities: [N, C] posterior p(c | s, t)
            log_likelihood: Total log-likelihood
        """
        # Log p(s, t | c) for each component
        log_prob = self._compute_log_likelihood_per_component(S, T)

        # Add log weights: log p(c)
        log_prob_weighted = log_prob + np.log(self.params.weights)

        # Log-sum-exp for normalization
        log_sum = logsumexp(log_prob_weighted, axis=1, keepdims=True)

        # Responsibilities: p(c | s, t)
        log_resp = log_prob_weighted - log_sum
        responsibilities = np.exp(log_resp)

        # Total log-likelihood
        log_likelihood = log_sum.sum()

        return responsibilities, log_likelihood

    def _m_step(self, S: np.ndarray, T: np.ndarray, responsibilities: np.ndarray):
        """M-step: update all parameters.

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]
            responsibilities: [N, C] posterior p(c | s, t)
        """
        N, D_s = S.shape
        D_t = T.shape[1]
        C = self.n_components

        # Sum of responsibilities per component
        N_c = responsibilities.sum(axis=0) + 1e-10  # [C]

        # Update weights
        self.params.weights = N_c / N

        for c in range(C):
            gamma_c = responsibilities[:, c]  # [N]

            # Update semantic mean
            self.params.semantic_means[c] = (gamma_c @ S) / N_c[c]

            # Centered semantic
            S_centered = S - self.params.semantic_means[c]  # [N, D_s]

            # Update semantic variance (diagonal)
            weighted_sq = gamma_c[:, None] * S_centered**2
            self.params.semantic_vars[c] = weighted_sq.sum(axis=0) / N_c[c] + self.reg_covar

            # Weighted least squares for regression: t = μₜ + A @ s_centered + noise
            # Normal equations: A = (Σ γ t s^T) @ (Σ γ s s^T)^{-1}
            # With diagonal assumption on s: (Σ γ s s^T) ≈ diag(Σ γ s²) = N_c * var_s

            # Simpler: solve separately for each textural dimension
            # For dimension d: A[d, :] = cov(t_d, s) / var(s) weighted by gamma

            # Weighted covariance: Σ γ (t - mean_t) @ (s - mean_s)^T / N_c
            # But we want intercept too, so we do proper weighted regression

            # Design matrix approach for each component
            # t_d = μₜ_d + A[d, :] @ s_centered + noise

            # Weighted mean of t
            T_weighted_mean = (gamma_c @ T) / N_c[c]  # [D_t]

            # Weighted cross-covariance
            # [D_t, D_s] = (γ * (T - mean_t))^T @ S_centered / N_c
            T_centered = T - T_weighted_mean
            cross_cov = (gamma_c[:, None] * T_centered).T @ S_centered / N_c[c]  # [D_t, D_s]

            # Regression coefficient: A = cross_cov @ diag(1/var_s)
            # Since var_s is diagonal
            self.params.regression_A[c] = cross_cov / self.params.semantic_vars[c]  # [D_t, D_s]

            # Update textural mean (intercept)
            # The intercept is the weighted mean of t (since we center s around its mean)
            self.params.textural_means[c] = T_weighted_mean

            # Residuals: t - μₜ - A @ s_centered
            predicted = S_centered @ self.params.regression_A[c].T  # [N, D_t]
            residuals = T - self.params.textural_means[c] - predicted  # [N, D_t]

            # Update residual variance (diagonal)
            weighted_resid_sq = gamma_c[:, None] * residuals**2
            self.params.residual_vars[c] = weighted_resid_sq.sum(axis=0) / N_c[c] + self.reg_covar

    def _e_step_batched(self, S: np.ndarray, T: np.ndarray, batch_size: int) -> Tuple[np.ndarray, float]:
        """E-step: loop over components, vectorize over samples.

        Memory efficient: creates [B, D] arrays, not [B, C, D].

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]
            batch_size: Process this many samples at a time

        Returns:
            responsibilities: [N, C] posterior p(c | s, t)
            log_likelihood: Total log-likelihood
        """
        N = S.shape[0]
        C = self.n_components

        # Pre-compute constants for each component
        log_weights = np.log(self.params.weights + 1e-300)  # [C]
        log_norm_s = -0.5 * np.sum(np.log(2 * np.pi * self.params.semantic_vars), axis=1)  # [C]
        log_norm_t = -0.5 * np.sum(np.log(2 * np.pi * self.params.residual_vars), axis=1)  # [C]

        # Pre-compute inverse variances for speed
        inv_sem_vars = 1.0 / self.params.semantic_vars  # [C, D_s]
        inv_res_vars = 1.0 / self.params.residual_vars  # [C, D_t]

        log_prob = np.zeros((N, C), dtype=np.float64)

        # Loop over components, vectorize over samples
        for c in range(C):
            # Semantic: diff_s = S - mu_s[c], shape [N, D_s]
            diff_s = S - self.params.semantic_means[c]

            # log p(s|c) = -0.5 * sum_d (diff_s^2 / var_s) + log_norm
            log_p_s = -0.5 * np.sum(diff_s**2 * inv_sem_vars[c], axis=1) + log_norm_s[c]

            # Conditional mean: mu_t[c] + A[c] @ diff_s.T -> [N, D_t]
            cond_mean = self.params.textural_means[c] + diff_s @ self.params.regression_A[c].T

            # Textural residual
            diff_t = T - cond_mean

            # log p(t|s,c)
            log_p_t = -0.5 * np.sum(diff_t**2 * inv_res_vars[c], axis=1) + log_norm_t[c]

            # Joint log prob for this component
            log_prob[:, c] = log_p_s + log_p_t + log_weights[c]

        # Normalize to get responsibilities
        log_sum = logsumexp(log_prob, axis=1, keepdims=True)
        responsibilities = np.exp(log_prob - log_sum).astype(np.float32)
        total_ll = log_sum.sum()

        return responsibilities, total_ll

    def _m_step_batched(self, S: np.ndarray, T: np.ndarray, responsibilities: np.ndarray, batch_size: int):
        """M-step: vectorized matrix operations, loop over components only where needed.

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]
            responsibilities: [N, C] posterior p(c | s, t)
            batch_size: Not used in this version (kept for API compatibility)
        """
        N, D_s = S.shape
        D_t = T.shape[1]
        C = self.n_components

        # Sufficient statistics for means - fully vectorized
        N_c = responsibilities.sum(axis=0).astype(np.float64)  # [C]
        N_c = np.maximum(N_c, 1e-10)

        # Weighted sums: [C, D] = [C, N] @ [N, D]
        sum_s = responsibilities.T @ S  # [C, D_s]
        sum_t = responsibilities.T @ T  # [C, D_t]

        # Update weights and means
        self.params.weights = (N_c / N).astype(np.float32)
        self.params.semantic_means = (sum_s / N_c[:, None]).astype(np.float32)
        self.params.textural_means = (sum_t / N_c[:, None]).astype(np.float32)

        # Variances and cross-covariance: loop over components
        for c in range(C):
            gamma_c = responsibilities[:, c]  # [N]

            # Centered data
            S_centered = S - self.params.semantic_means[c]  # [N, D_s]
            T_centered = T - self.params.textural_means[c]  # [N, D_t]

            # Weighted variance (diagonal): sum_n gamma[n] * s[n,d]^2 / N_c
            self.params.semantic_vars[c] = ((gamma_c @ (S_centered**2)) / N_c[c] + self.reg_covar).astype(np.float32)

            # Cross-covariance: [D_t, D_s] = [D_t, N] @ [N, D_s] / N_c
            # = (gamma * T_centered).T @ S_centered / N_c
            cross_cov = (gamma_c[:, None] * T_centered).T @ S_centered / N_c[c]

            # Regression: A = cross_cov / var_s (element-wise division by diagonal)
            self.params.regression_A[c] = (cross_cov / self.params.semantic_vars[c]).astype(np.float32)

            # Residual variance
            predicted = S_centered @ self.params.regression_A[c].T  # [N, D_t]
            residuals = T_centered - predicted  # [N, D_t]
            self.params.residual_vars[c] = ((gamma_c @ (residuals**2)) / N_c[c] + self.reg_covar).astype(np.float32)

    def fit(self, S: np.ndarray, T: np.ndarray, batch_size: int = 50000) -> 'MixtureOfLinearRegressions':
        """Fit the model using mini-batch EM algorithm.

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]
            batch_size: Size of mini-batches for E-step (memory optimization)

        Returns:
            self
        """
        import time

        N, D_s = S.shape
        D_t = T.shape[1]

        if self.verbose:
            print(f"Fitting Mixture of Linear Regressions")
            print(f"  N={N:,}, D_s={D_s}, D_t={D_t}, C={self.n_components}")
            print(f"  batch_size={batch_size:,}")

        # Initialize
        self._initialize_parameters(S, T)

        # EM iterations
        prev_log_likelihood = -np.inf
        self.log_likelihood_history_ = []

        for iteration in range(self.max_iter):
            iter_start = time.time()

            # E-step (mini-batch for memory efficiency)
            e_start = time.time()
            responsibilities, log_likelihood = self._e_step_batched(S, T, batch_size)
            e_time = time.time() - e_start

            self.log_likelihood_history_.append(log_likelihood)

            # Check convergence
            rel_change = 0.0
            if iteration > 0:
                rel_change = (log_likelihood - prev_log_likelihood) / (abs(prev_log_likelihood) + 1e-10)

            # M-step
            m_start = time.time()
            self._m_step_batched(S, T, responsibilities, batch_size)
            m_time = time.time() - m_start

            iter_time = time.time() - iter_start

            # Always print progress
            if self.verbose:
                avg_ll = log_likelihood / N
                print(f"  Iter {iteration:3d}: LL={log_likelihood:.2e}, avg_LL={avg_ll:.4f}, "
                      f"rel_change={rel_change:+.2e}, E={e_time:.1f}s, M={m_time:.1f}s, total={iter_time:.1f}s")

            if iteration > 0:
                if rel_change < self.tol and rel_change >= 0:
                    self.converged_ = True
                    if self.verbose:
                        print(f"  Converged at iteration {iteration}")
                    break

                if rel_change < -0.01:  # Only warn for significant decreases
                    if self.verbose:
                        print(f"  Warning: significant log-likelihood decrease!")

            prev_log_likelihood = log_likelihood

        self.n_iter_ = iteration + 1

        if self.verbose:
            if not self.converged_:
                print(f"  Did not converge after {self.max_iter} iterations")
            print(f"  Final LL={self.log_likelihood_history_[-1]:.2f}")

        return self

    def compute_responsibilities(self, S: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute posterior responsibilities p(c | s, t).

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]

        Returns:
            responsibilities [N, C]
        """
        responsibilities, _ = self._e_step(S, T)
        return responsibilities

    def compute_residuals(self, S: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute textural residuals for the most likely component.

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]

        Returns:
            residuals [N, D_t] for most likely component assignment
        """
        responsibilities = self.compute_responsibilities(S, T)
        assignments = responsibilities.argmax(axis=1)

        residuals = np.zeros_like(T)
        for c in range(self.n_components):
            mask = assignments == c
            if mask.sum() == 0:
                continue
            S_c = S[mask]
            T_c = T[mask]
            S_centered = S_c - self.params.semantic_means[c]
            predicted = S_centered @ self.params.regression_A[c].T + self.params.textural_means[c]
            residuals[mask] = T_c - predicted

        return residuals

    def score(self, S: np.ndarray, T: np.ndarray) -> float:
        """Compute average log-likelihood.

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]

        Returns:
            Average log-likelihood per sample
        """
        _, log_likelihood = self._e_step(S, T)
        return log_likelihood / S.shape[0]

    def get_parameters(self) -> MLRParameters:
        """Get fitted parameters."""
        if self.params is None:
            raise ValueError("Model not fitted yet")
        return self.params

    def save(self, path: Path):
        """Save the fitted model."""
        if self.params is None:
            raise ValueError("Model not fitted yet")
        self.params.save(path)

    def load(self, path: Path):
        """Load a fitted model."""
        self.params = MLRParameters.load(path)
        self.n_components = self.params.n_components

    def plot_diagnostics(self, S: np.ndarray, T: np.ndarray, save_path: Optional[Path] = None):
        """Plot diagnostic plots for fit quality.

        Args:
            S: Semantic features [N, D_s]
            T: Textural features [N, D_t]
            save_path: If provided, save figure to this path
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Log-likelihood convergence
        ax = axes[0, 0]
        ax.plot(self.log_likelihood_history_)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-likelihood')
        ax.set_title('EM Convergence')
        ax.grid(True)

        # 2. Component weights
        ax = axes[0, 1]
        ax.bar(range(self.n_components), self.params.weights)
        ax.set_xlabel('Component')
        ax.set_ylabel('Weight')
        ax.set_title('Mixture Weights')

        # 3. Component assignment histogram
        ax = axes[0, 2]
        responsibilities = self.compute_responsibilities(S, T)
        assignments = responsibilities.argmax(axis=1)
        ax.hist(assignments, bins=self.n_components, edgecolor='black')
        ax.set_xlabel('Component')
        ax.set_ylabel('Count')
        ax.set_title('Hard Assignment Distribution')

        # 4. Residual variance by component
        ax = axes[1, 0]
        mean_resid_var = self.params.residual_vars.mean(axis=1)
        ax.bar(range(self.n_components), mean_resid_var)
        ax.set_xlabel('Component')
        ax.set_ylabel('Mean Residual Variance')
        ax.set_title('Residual Variance per Component')

        # 5. Regression strength (Frobenius norm of A)
        ax = axes[1, 1]
        A_norms = np.linalg.norm(self.params.regression_A, axis=(1, 2))
        ax.bar(range(self.n_components), A_norms)
        ax.set_xlabel('Component')
        ax.set_ylabel('||A||_F')
        ax.set_title('Regression Matrix Norm per Component')

        # 6. R² per component (variance explained by regression)
        ax = axes[1, 2]
        r_squared = []
        for c in range(self.n_components):
            mask = assignments == c
            if mask.sum() < 2:
                r_squared.append(0)
                continue
            S_c = S[mask]
            T_c = T[mask]
            S_centered = S_c - self.params.semantic_means[c]
            predicted = S_centered @ self.params.regression_A[c].T + self.params.textural_means[c]
            ss_res = ((T_c - predicted)**2).sum()
            ss_tot = ((T_c - T_c.mean(axis=0))**2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            r_squared.append(max(0, r2))  # Clip negative R² to 0

        ax.bar(range(self.n_components), r_squared)
        ax.set_xlabel('Component')
        ax.set_ylabel('R²')
        ax.set_title('Variance Explained by Regression')
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Diagnostics saved to {save_path}")

        plt.show()

        # Print summary statistics
        print("\n" + "=" * 60)
        print("FIT SUMMARY")
        print("=" * 60)
        print(f"Components: {self.n_components}")
        print(f"Converged: {self.converged_} (iterations: {self.n_iter_})")
        print(f"Final log-likelihood: {self.log_likelihood_history_[-1]:.2f}")
        print(f"Avg log-likelihood per sample: {self.log_likelihood_history_[-1] / S.shape[0]:.4f}")
        print(f"\nComponent weight range: [{self.params.weights.min():.4f}, {self.params.weights.max():.4f}]")
        print(f"Mean R² across components: {np.mean(r_squared):.4f}")
        print(f"Mean ||A||_F across components: {A_norms.mean():.2f}")

        # Effective number of components (perplexity of weights)
        eff_components = np.exp(-np.sum(self.params.weights * np.log(self.params.weights + 1e-10)))
        print(f"Effective number of components: {eff_components:.1f}")
