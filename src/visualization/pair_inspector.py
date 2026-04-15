"""Detection pair inspector: view image pairs by distance range."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage


def show_pair(
    i_global: int,
    j_global: int,
    dist_val: float,
    same_vp: bool,
    valid_dets: list[str],
    raw_vps: np.ndarray,
    coarse_vps: np.ndarray,
    visibility: np.ndarray,
    edge_proximity: np.ndarray,
    eigenvectors: np.ndarray,
    dataset,
    det_to_annot: Dict[str, object] | None = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Display two detection images side by side with metadata.

    Args:
        i_global, j_global: Indices into valid_dets / raw_vps / etc.
        dist_val: The Laplacian distance between the pair.
        same_vp: Whether the pair is same-VP (relaxed).
        valid_dets: List of detection IDs.
        raw_vps: [N] raw viewpoint labels.
        coarse_vps: [N] coarse viewpoint labels.
        visibility: [N] visibility scores.
        edge_proximity: [N] edge proximity scores.
        eigenvectors: [N, n_eig] eigenvector matrix.
        dataset: Dataset with .get_detection(det_id) method.
        det_to_annot: Optional dict det_id -> annotation for identity info.
        figsize: Figure size.

    Returns:
        Figure.
    """
    def _angle1(gi):
        return np.degrees(np.arctan2(eigenvectors[gi, 2], eigenvectors[gi, 1]))

    def _angle2(gi):
        return np.degrees(np.arctan2(eigenvectors[gi, 3], eigenvectors[gi, 2]))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, gi, label in [(axes[0], i_global, "A"), (axes[1], j_global, "B")]:
        did = valid_dets[gi]
        det_obj = dataset.get_detection(did)
        img = PILImage.open(det_obj.image_path).convert("RGB")
        img_w, img_h = img.size
        sx1, sy1, sx2, sy2 = det_obj.bbox.int().tolist()
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(img_w, sx2), min(img_h, sy2)
        crop = img.crop((sx1, sy1, sx2, sy2))
        ax.imshow(crop)

        ind = "?"
        if det_to_annot and did in det_to_annot:
            ind = str(det_to_annot[did].individual_id)[:18]

        ax.set_title(
            f"{label}: {raw_vps[gi]} (coarse: {coarse_vps[gi]})\n"
            f"vis={visibility[gi]:.2f}, edge={edge_proximity[gi]:.3f}\n"
            f"θ1={_angle1(gi):+.1f}°, θ2={_angle2(gi):+.1f}°\n"
            f"ind={ind}",
            fontsize=11,
        )
        ax.axis("off")

    color = "green" if same_vp else "red"
    relation = "SAME" if same_vp else "DIFF"
    fig.suptitle(f"d={dist_val:.4f}, {relation} VP", fontsize=14, y=1.02, color=color)
    plt.tight_layout()
    return fig
