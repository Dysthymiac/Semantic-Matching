"""Eigendecomposition of graph Laplacians."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import eigsh


def smallest_eigenvectors(
    L: spmatrix,
    n_eig: int,
    sigma: float = 1e-8,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the n_eig smallest eigenvectors of a sparse symmetric matrix
    using shift-invert Lanczos (much faster than which='SM').

    Args:
        L: Sparse symmetric matrix (e.g. normalized Laplacian).
        n_eig: Number of eigenpairs to compute.
        sigma: Shift for shift-invert mode. Use a small positive value
            to avoid singularity at the zero eigenvalue.
        verbose: Print timing and eigenvalue summary.

    Returns:
        (eigenvalues, eigenvectors) sorted by increasing eigenvalue.
        eigenvalues: [n_eig], eigenvectors: [N, n_eig].
    """
    t0 = time.time()
    eigenvalues, eigenvectors = eigsh(L, k=n_eig, sigma=sigma, which="LM")
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if verbose:
        elapsed = time.time() - t0
        print(
            f"  eigsh {n_eig} smallest: {elapsed:.1f}s, "
            f"eigvals[0]={eigenvalues[0]:.2e}, eigvals[-1]={eigenvalues[-1]:.4f}"
        )
    return eigenvalues, eigenvectors
