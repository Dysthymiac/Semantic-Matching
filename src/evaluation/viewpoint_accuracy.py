"""Viewpoint classification accuracy with coarse mapping and relaxed evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np

# Coarse viewpoint mapping: raw fine-grained labels -> 4 cardinal directions.
COARSE_MAP: Dict[str, str] = {
    "right": "right",
    "backright": "right",
    "frontright": "right",
    "upright": "right",
    "downright": "right",
    "upbackright": "right",
    "downbackright": "right",
    "upfrontright": "right",
    "left": "left",
    "backleft": "left",
    "frontleft": "left",
    "upleft": "left",
    "downleft": "left",
    "upbackleft": "left",
    "upfrontleft": "left",
    "front": "front",
    "back": "back",
}


def coarse_viewpoint(raw_vp: str) -> str | None:
    """Map a raw fine-grained viewpoint to its coarse direction.

    Returns None if the viewpoint is not in the mapping.
    """
    return COARSE_MAP.get(raw_vp)


def relaxed_correct(raw_vp: str, pred_vp: str) -> bool:
    """Check if a prediction is relaxed-correct for a raw viewpoint.

    A prediction counts as correct if:
    - It matches the coarse mapping of the raw viewpoint, OR
    - The raw viewpoint is compound (e.g. 'frontright') and the
      prediction matches either component ('front' or 'right').

    Args:
        raw_vp: The fine-grained ground-truth viewpoint label.
        pred_vp: The predicted coarse viewpoint.

    Returns:
        True if the prediction is relaxed-correct.
    """
    coarse_true = COARSE_MAP.get(raw_vp, raw_vp)
    if pred_vp == coarse_true:
        return True
    for pure_vp in ("right", "left", "front", "back"):
        if pure_vp in raw_vp and pred_vp == pure_vp:
            return True
    return False


def relaxed_same_viewpoint(raw_a: str, raw_b: str) -> bool:
    """Check if two raw viewpoints are relaxed-same.

    Two viewpoints are relaxed-same if they share the same coarse mapping
    OR share any pure component (e.g. 'frontright' and 'right' share 'right').

    Args:
        raw_a, raw_b: Fine-grained viewpoint labels.

    Returns:
        True if they are relaxed-same.
    """
    coarse_a = COARSE_MAP.get(raw_a, raw_a)
    coarse_b = COARSE_MAP.get(raw_b, raw_b)
    if coarse_a == coarse_b:
        return True
    for pure in ("right", "left", "front", "back"):
        if pure in raw_a and pure in raw_b:
            return True
    return False
