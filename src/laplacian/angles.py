"""Angle recovery from Laplacian eigenvectors."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def angle_pair(
    eigenvectors: np.ndarray,
    ev_a: int,
    ev_b: int,
) -> np.ndarray:
    """Compute atan2(EV_b, EV_a) for each point.

    Args:
        eigenvectors: [N, n_eig] eigenvector matrix.
        ev_a: Column index for the cosine-like component.
        ev_b: Column index for the sine-like component.

    Returns:
        Angles in radians, shape [N].
    """
    return np.arctan2(eigenvectors[:, ev_b], eigenvectors[:, ev_a])


def angles_from_evs(
    eigenvectors: np.ndarray,
    pairs: tuple[tuple[int, int], ...] = ((1, 2), (2, 3)),
) -> list[np.ndarray]:
    """Compute multiple angular coordinates from eigenvector pairs.

    Default pairs (1,2) and (2,3) give the two viewpoint angles
    from the clean 3D subspace (EVs 1-3).

    Args:
        eigenvectors: [N, n_eig] eigenvector matrix.
        pairs: Tuples of (cos_ev_idx, sin_ev_idx).

    Returns:
        List of angle arrays, each [N] in radians.
    """
    return [angle_pair(eigenvectors, a, b) for a, b in pairs]
