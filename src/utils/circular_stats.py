"""Circular (angular) statistics utilities."""

from __future__ import annotations

import numpy as np


def circular_mean(angles: np.ndarray) -> float:
    """Compute the circular (directional) mean of angles in radians."""
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def circular_std(angles: np.ndarray) -> float:
    """Compute the circular standard deviation of angles in radians.

    Based on the mean resultant length R: std = sqrt(-2 * ln(R)).
    """
    R = np.sqrt(np.sin(angles).mean() ** 2 + np.cos(angles).mean() ** 2)
    return float(np.sqrt(-2.0 * np.log(max(R, 1e-10))))


def circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the shortest angular distance between two arrays of angles (radians).

    Returns values in [0, pi].
    """
    d = np.abs(a - b)
    return np.minimum(d, 2 * np.pi - d)


def circular_midpoint(a: float, b: float) -> float:
    """Compute the circular midpoint of two angles in radians."""
    return float(
        np.arctan2(
            (np.sin(a) + np.sin(b)) / 2.0,
            (np.cos(a) + np.cos(b)) / 2.0,
        )
    )
