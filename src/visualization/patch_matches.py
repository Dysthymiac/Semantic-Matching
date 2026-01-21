"""Visualization of patch-level matches between detections."""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

from ..matching.patch_matching import PatchMatch
from .primitives import get_crop_bounds, compute_padding_info


def patch_grid_to_crop_center(
    row: int,
    col: int,
    crop_w: int,
    crop_h: int,
    target_size: int = 512,
    patch_size: int = 16,
) -> Optional[Tuple[float, float]]:
    """
    Convert patch grid coordinates to center point in crop space.

    Args:
        row, col: Patch grid coordinates
        crop_w, crop_h: Actual crop dimensions
        target_size: Size used for DINOv3 input (512)
        patch_size: DINOv3 patch size (16)

    Returns:
        (x, y) center coordinates in crop space, or None if patch is in padding
    """
    scale, pad_left, pad_top = compute_padding_info(crop_w, crop_h, target_size)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    # Patch center in 512 space
    p_cx = col * patch_size + patch_size / 2
    p_cy = row * patch_size + patch_size / 2

    # Content bounds in 512 space
    c_x1, c_y1 = pad_left, pad_top
    c_x2, c_y2 = pad_left + new_w, pad_top + new_h

    # Check if patch center is within content
    if p_cx < c_x1 or p_cx > c_x2 or p_cy < c_y1 or p_cy > c_y2:
        return None

    # Map to crop space
    x = (p_cx - c_x1) / scale
    y = (p_cy - c_y1) / scale

    return (x, y)


def draw_patch_matches(
    ax,
    crop1: Image.Image,
    crop2: Image.Image,
    matches: List[PatchMatch],
    crop1_size: Tuple[int, int],
    crop2_size: Tuple[int, int],
    gap: int = 20,
    cmap: str = 'RdYlGn',
    linewidth_range: Tuple[float, float] = (0.5, 3.0),
    alpha_range: Tuple[float, float] = (0.3, 1.0),
) -> None:
    """
    Draw two crops side by side with lines connecting matched patches.

    Args:
        ax: Matplotlib axes
        crop1, crop2: PIL Images of the two crops
        matches: List of PatchMatch objects
        crop1_size, crop2_size: (width, height) of crops before any resizing
        gap: Pixel gap between the two images
        cmap: Colormap for similarity coloring (green=high, red=low)
        linewidth_range: (min, max) line width based on similarity
        alpha_range: (min, max) alpha based on similarity
    """
    crop1_w, crop1_h = crop1_size
    crop2_w, crop2_h = crop2_size

    # Create combined image
    combined_w = crop1.width + gap + crop2.width
    combined_h = max(crop1.height, crop2.height)
    combined = Image.new('RGB', (combined_w, combined_h), (255, 255, 255))
    combined.paste(crop1, (0, 0))
    combined.paste(crop2, (crop1.width + gap, 0))

    ax.imshow(combined)

    if not matches:
        return

    # Get similarity range for normalization
    sims = [m.similarity for m in matches]
    sim_min, sim_max = min(sims), max(sims)
    sim_range = sim_max - sim_min if sim_max > sim_min else 1.0

    # Get colormap
    colormap = plt.cm.get_cmap(cmap)

    # Build line segments
    lines = []
    colors = []
    linewidths = []
    alphas = []

    for match in matches:
        # Get center coordinates in crop space
        center1 = patch_grid_to_crop_center(
            match.coord1[0], match.coord1[1], crop1_w, crop1_h
        )
        center2 = patch_grid_to_crop_center(
            match.coord2[0], match.coord2[1], crop2_w, crop2_h
        )

        if center1 is None or center2 is None:
            continue

        # Offset second point by crop1 width + gap
        x1, y1 = center1
        x2, y2 = center2[0] + crop1.width + gap, center2[1]

        lines.append([(x1, y1), (x2, y2)])

        # Normalize similarity to [0, 1]
        norm_sim = (match.similarity - sim_min) / sim_range

        # Map to color
        colors.append(colormap(norm_sim))

        # Map to linewidth
        lw_min, lw_max = linewidth_range
        linewidths.append(lw_min + norm_sim * (lw_max - lw_min))

        # Map to alpha
        a_min, a_max = alpha_range
        alphas.append(a_min + norm_sim * (a_max - a_min))

    # Draw all lines using LineCollection for efficiency
    lc = LineCollection(lines, colors=colors, linewidths=linewidths, alpha=0.7)
    ax.add_collection(lc)


def visualize_patch_matches(
    det1,
    det2,
    matches: List[PatchMatch],
    title: str = "",
    figsize: Tuple[int, int] = (16, 8),
    gap: int = 20,
) -> plt.Figure:
    """
    Visualize patch matches between two detections.

    Args:
        det1, det2: Detection objects with image_path, square_crop_bbox, features, patch_mask
        matches: List of PatchMatch objects
        title: Figure title
        figsize: Figure size
        gap: Gap between images

    Returns:
        matplotlib Figure
    """
    # Load and crop images
    img1 = Image.open(det1.image_path).convert('RGB')
    img2 = Image.open(det2.image_path).convert('RGB')

    img1_w, img1_h = img1.size
    img2_w, img2_h = img2.size

    x1_1, y1_1, x2_1, y2_1 = get_crop_bounds(det1.square_crop_bbox, img1_w, img1_h)
    x1_2, y1_2, x2_2, y2_2 = get_crop_bounds(det2.square_crop_bbox, img2_w, img2_h)

    crop1 = img1.crop((x1_1, y1_1, x2_1, y2_1))
    crop2 = img2.crop((x1_2, y1_2, x2_2, y2_2))

    crop1_size = (x2_1 - x1_1, y2_1 - y1_1)
    crop2_size = (x2_2 - x1_2, y2_2 - y1_2)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    draw_patch_matches(
        ax, crop1, crop2, matches,
        crop1_size, crop2_size,
        gap=gap,
    )

    n_matches = len(matches)
    if matches:
        avg_sim = np.mean([m.similarity for m in matches])
        ax.set_title(f"{title}\n{n_matches} matches, avg similarity: {avg_sim:.3f}")
    else:
        ax.set_title(f"{title}\nNo matches")

    ax.set_xticks([])
    ax.set_yticks([])

    return fig
