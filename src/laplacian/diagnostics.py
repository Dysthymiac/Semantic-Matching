"""Diagnostic tools for Laplacian eigenvectors.

Detects nuisance-hijacked eigenvectors (bottleneck / Fiedler-like modes)
by checking for bimodal or heavy-tailed distributions.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def ev_health_check(
    eigenvectors: np.ndarray,
    ev_indices: list[int] | None = None,
    mad_z_threshold: float = 20.0,
) -> Dict[int, Dict]:
    """Check each eigenvector for bottleneck signatures.

    A bimodal / heavy-tailed EV distribution (high MAD-z outliers) indicates
    a Fiedler-like vector encoding a graph bottleneck rather than smooth
    manifold structure.

    Args:
        eigenvectors: [N, n_eig] eigenvector matrix.
        ev_indices: Which column indices to check. None = all columns.
        mad_z_threshold: Threshold for flagging extreme outliers.

    Returns:
        Dict mapping EV index -> {
            'median', 'mad', 'max_abs_z', 'n_outliers', 'is_bottleneck',
            'min', 'max', 'asymmetry' (max/min ratio of range around median)
        }
    """
    if ev_indices is None:
        ev_indices = list(range(eigenvectors.shape[1]))

    results = {}
    for i in ev_indices:
        vals = eigenvectors[:, i]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med)) + 1e-12
        z = np.abs(vals - med) / mad
        n_outliers = int((z > mad_z_threshold).sum())
        max_abs_z = float(z.max())

        # Asymmetry: ratio of range above vs below median
        range_above = vals.max() - med
        range_below = med - vals.min()
        asymmetry = max(range_above, range_below) / max(min(range_above, range_below), 1e-12)

        is_bottleneck = (n_outliers > 0 and max_abs_z > mad_z_threshold * 2) or asymmetry > 20

        results[i] = {
            "median": float(med),
            "mad": float(mad),
            "max_abs_z": max_abs_z,
            "n_outliers": n_outliers,
            "is_bottleneck": is_bottleneck,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "asymmetry": float(asymmetry),
        }
    return results


def find_bottleneck_evs(
    eigenvectors: np.ndarray,
    ev_indices: list[int] | None = None,
    mad_z_threshold: float = 20.0,
) -> List[int]:
    """Return indices of eigenvectors flagged as bottleneck modes.

    Args:
        eigenvectors: [N, n_eig] eigenvector matrix.
        ev_indices: Which columns to check. None = all.
        mad_z_threshold: MAD-z threshold for outlier detection.

    Returns:
        List of EV column indices that are likely nuisance bottlenecks.
    """
    health = ev_health_check(eigenvectors, ev_indices, mad_z_threshold)
    return [i for i, info in health.items() if info["is_bottleneck"]]
