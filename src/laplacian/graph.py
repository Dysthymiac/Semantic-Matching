"""kNN graph construction and normalized Laplacian."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    X: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
) -> Tuple[csr_matrix, np.ndarray]:
    """Build a symmetric kNN graph.

    Args:
        X: Feature matrix [N, D].
        k: Number of nearest neighbors.
        metric: Distance metric for NearestNeighbors.

    Returns:
        (A, degrees) where A is the sparse symmetric adjacency matrix
        and degrees is [N] node degree array.
    """
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    N = len(X)
    rows, cols = [], []
    for i in range(N):
        for jj in range(1, k):  # skip self at index 0
            rows.extend([i, indices[i, jj]])
            cols.extend([indices[i, jj], i])

    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A = (A > 0).astype(np.float64)  # dedupe mutual edges
    degrees = np.asarray(A.sum(axis=1)).ravel()
    return A, degrees


def normalized_laplacian(
    A: csr_matrix,
    degrees: np.ndarray,
) -> csr_matrix:
    """Compute the symmetric normalized Laplacian L_sym = D^{-1/2} (D - A) D^{-1/2}.

    Args:
        A: Sparse symmetric adjacency matrix.
        degrees: Node degree array [N].

    Returns:
        L_sym as a sparse matrix.
    """
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
    D_sparse = diags(degrees)
    return D_inv_sqrt @ (D_sparse - A) @ D_inv_sqrt
