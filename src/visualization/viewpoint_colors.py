"""Viewpoint color palettes for consistent plotting across notebooks."""

from __future__ import annotations

from typing import Dict

# 8 fine-grained viewpoints — semantically shaded:
#   right family = blues, left family = reds, back = orange, front = green.
VP_COLORS_8: Dict[str, str] = {
    "right": "royalblue",
    "backright": "navy",
    "frontright": "deepskyblue",
    "left": "red",
    "backleft": "darkred",
    "frontleft": "hotpink",
    "back": "orange",
    "front": "limegreen",
}

# 4 coarse viewpoints.
VP_COLORS_4: Dict[str, str] = {
    "right": "blue",
    "left": "red",
    "front": "green",
    "back": "orange",
}

# RGB tuples for interactive viewers (deck.gl, Leaflet, etc.)
VP_COLORS_RGB: Dict[str, list[int]] = {
    "right": [30, 120, 180],
    "backright": [100, 150, 220],
    "frontright": [0, 190, 220],
    "left": [220, 40, 40],
    "backleft": [250, 130, 130],
    "frontleft": [180, 40, 40],
    "front": [40, 160, 40],
    "back": [255, 165, 0],
    "unknown": [80, 80, 80],
}
