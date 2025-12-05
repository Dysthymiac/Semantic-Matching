"""Low-level visualization primitives."""

from typing import List, Tuple, Optional
import numpy as np
import torch
from matplotlib.patches import Rectangle


def draw_bbox(ax, bbox: torch.Tensor, color: str = 'red', linewidth: int = 2) -> None:
    """Draw bounding box on axes. bbox is [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox.int().tolist()
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                            fill=False, edgecolor=color, linewidth=linewidth))


def compute_padding_info(crop_w: int, crop_h: int, target_size: int = 512) -> Tuple[float, int, int]:
    """
    Compute padding info from resize_and_pad_image.

    Returns (scale, pad_left, pad_top) where:
    - scale: factor used to resize crop to fit in target_size
    - pad_left, pad_top: padding added to center content in target_size x target_size
    """
    scale = min(target_size / crop_w, target_size / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    return scale, pad_left, pad_top


def patch_to_crop_coords(
    patch_i: int, patch_j: int,
    crop_w: int, crop_h: int,
    target_size: int = 512,
    patch_size: int = 16
) -> Optional[Tuple[float, float, float, float]]:
    """
    Map single patch (i, j) from 512x512 space to crop coordinates.

    Returns (x, y, width, height) in crop space, or None if patch is in padding.
    """
    scale, pad_left, pad_top = compute_padding_info(crop_w, crop_h, target_size)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    # Patch bounds in 512 space
    p_x1, p_y1 = patch_j * patch_size, patch_i * patch_size
    p_x2, p_y2 = p_x1 + patch_size, p_y1 + patch_size

    # Content bounds in 512 space
    c_x1, c_y1 = pad_left, pad_top
    c_x2, c_y2 = pad_left + new_w, pad_top + new_h

    # Check if patch overlaps content
    if p_x2 <= c_x1 or p_x1 >= c_x2 or p_y2 <= c_y1 or p_y1 >= c_y2:
        return None

    # Clip to content and map to crop space
    x1 = (max(p_x1, c_x1) - c_x1) / scale
    y1 = (max(p_y1, c_y1) - c_y1) / scale
    x2 = (min(p_x2, c_x2) - c_x1) / scale
    y2 = (min(p_y2, c_y2) - c_y1) / scale

    return (x1, y1, x2 - x1, y2 - y1)


def get_crop_bounds(square_crop_bbox: torch.Tensor, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Get actual crop bounds by clamping square_crop_bbox to image."""
    x1, y1, x2, y2 = square_crop_bbox.int().tolist()
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def patch_coords_in_crop(
    patch_mask: torch.Tensor,
    crop_w: int, crop_h: int,
    target_size: int = 512,
    patch_size: int = 16
) -> List[Tuple[float, float, float, float]]:
    """
    Convert patch mask to rectangle coordinates in crop space.

    Returns list of (x, y, width, height) for each valid patch.
    """
    h_patches, w_patches = patch_mask.shape
    coords = []
    for i in range(h_patches):
        for j in range(w_patches):
            if patch_mask[i, j]:
                rect = patch_to_crop_coords(i, j, crop_w, crop_h, target_size, patch_size)
                if rect:
                    coords.append(rect)
    return coords


def draw_patches(
    ax,
    coords: List[Tuple[float, float, float, float]],
    colors: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    edgecolor: str = 'green',
    default_facecolor: str = 'green'
) -> None:
    """
    Draw patch rectangles on axes.

    Args:
        coords: List of (x, y, width, height) from patch_coords_in_crop
        colors: Optional RGB array [N, 3] for each patch, values in [0, 1]
    """
    for idx, (x, y, w, h) in enumerate(coords):
        facecolor = colors[idx] if colors is not None else default_facecolor
        ax.add_patch(Rectangle((x, y), w, h,
                                facecolor=facecolor, edgecolor=edgecolor,
                                alpha=alpha, linewidth=0.5))
