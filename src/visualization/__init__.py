"""Visualization utilities for detection analysis."""

from .primitives import (
    draw_bbox,
    draw_patches,
    patch_coords_in_crop,
    get_crop_bounds,
    compute_padding_info,
    patch_to_crop_coords,
)
from .gmm_vis import (
    compute_patch_responsibilities,
    responsibilities_to_colors,
)
