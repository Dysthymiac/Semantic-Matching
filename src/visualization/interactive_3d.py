"""Interactive 3D scatter plots using Plotly."""

from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np

from .viewpoint_colors import VP_COLORS_8


def plot_3d_eigenvectors(
    eigenvectors: np.ndarray,
    raw_vps: np.ndarray,
    ev_indices: tuple[int, int, int] = (1, 2, 3),
    colors: Dict[str, str] | None = None,
    max_points: int = 6000,
    seed: int = 42,
    marker_size: int = 3,
    opacity: float = 0.7,
    width: int = 950,
    height: int = 750,
    title: str | None = None,
):
    """Interactive 3D scatter of Laplacian eigenvectors via Plotly.

    Args:
        eigenvectors: [N, n_eig] eigenvector matrix.
        raw_vps: [N] fine-grained viewpoint labels.
        ev_indices: Which 3 EV columns to use for (x, y, z).
        colors: VP -> color. Defaults to VP_COLORS_8.
        max_points: Subsample for responsiveness.
        seed: Random seed for subsampling.
        marker_size: Plotly marker size.
        opacity: Marker opacity.
        width, height: Figure dimensions.
        title: Plot title. Auto-generated if None.

    Returns:
        Plotly Figure (call .show() to display).
    """
    import plotly.graph_objects as go

    if colors is None:
        colors = VP_COLORS_8

    N = len(raw_vps)
    if N > max_points:
        rng = np.random.RandomState(seed)
        sub = rng.choice(N, max_points, replace=False)
    else:
        sub = np.arange(N)

    counts = Counter(raw_vps)
    vp_sorted = [v for v, _ in counts.most_common() if counts[v] >= 5]
    ei, ej, ek = ev_indices

    if title is None:
        title = f"Laplacian EVs {ei},{ej},{ek} (n={len(sub)} subsample)"

    fig = go.Figure()
    for vp in vp_sorted:
        m = raw_vps[sub] == vp
        if m.sum() == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=eigenvectors[sub, ei][m],
            y=eigenvectors[sub, ej][m],
            z=eigenvectors[sub, ek][m],
            mode="markers",
            marker=dict(size=marker_size, color=colors.get(vp, "gray"), opacity=opacity),
            name=f"{vp} ({m.sum()})",
            hovertemplate=(
                f"{vp}<br>"
                f"EV{ei}=%{{x:.4f}}<br>"
                f"EV{ej}=%{{y:.4f}}<br>"
                f"EV{ek}=%{{z:.4f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"EV{ei}",
            yaxis_title=f"EV{ej}",
            zaxis_title=f"EV{ek}",
            aspectmode="cube",
        ),
        width=width,
        height=height,
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
