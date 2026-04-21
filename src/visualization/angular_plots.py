"""Angular viewpoint space visualizations."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde

from .viewpoint_colors import VP_COLORS_8


def plot_2d_angular_scatter(
    theta1_deg: np.ndarray,
    theta2_deg: np.ndarray,
    raw_vps: np.ndarray,
    colors: Dict[str, str] | None = None,
    marker_size: float = 25,
    alpha: float = 0.55,
    show_means: bool = True,
    title: str = "2D angular viewpoint space",
    figsize: tuple = (13, 11),
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Scatter all detections in the 2D angular viewpoint space.

    Args:
        theta1_deg: Angle 1 in degrees [N].
        theta2_deg: Angle 2 in degrees [N].
        raw_vps: Fine-grained viewpoint labels [N].
        colors: VP -> matplotlib color. Defaults to VP_COLORS_8.
        marker_size: Scatter point size.
        alpha: Scatter alpha.
        show_means: Overlay circular-mean X markers per class.
        title: Plot title.
        figsize: Figure size (ignored if ax is provided).
        ax: Existing axes to draw on. If None, creates a new figure.

    Returns:
        Figure if ax was None, else None.
    """
    if colors is None:
        colors = VP_COLORS_8
    counts = Counter(raw_vps)
    vp_sorted = [v for v, _ in counts.most_common() if counts[v] >= 5]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for vp in vp_sorted:
        m = raw_vps == vp
        ax.scatter(
            theta1_deg[m], theta2_deg[m],
            c=colors.get(vp, "gray"), s=marker_size, alpha=alpha,
            label=f"{vp} ({m.sum()})", edgecolors="none",
        )

    if show_means:
        for vp in vp_sorted:
            m = raw_vps == vp
            t1_mean = np.degrees(np.arctan2(
                np.sin(np.radians(theta1_deg[m])).mean(),
                np.cos(np.radians(theta1_deg[m])).mean(),
            ))
            t2_mean = np.degrees(np.arctan2(
                np.sin(np.radians(theta2_deg[m])).mean(),
                np.cos(np.radians(theta2_deg[m])).mean(),
            ))
            ax.scatter(
                t1_mean, t2_mean, marker="X", s=300,
                c=colors.get(vp, "gray"), edgecolors="black",
                linewidths=2, zorder=10,
            )

    ax.set_xlabel("Angle 1: atan2(EV2, EV1) [degrees]")
    ax.set_ylabel("Angle 2: atan2(EV3, EV2) [degrees]")
    ax.set_title(title)
    ax.legend(fontsize=10, ncol=2, loc="lower right", markerscale=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)

    if fig is not None:
        plt.tight_layout()
    return fig


def plot_per_vp_kde(
    theta1_deg: np.ndarray,
    theta2_deg: np.ndarray,
    raw_vps: np.ndarray,
    colors: Dict[str, str] | None = None,
    kde_bw: float = 0.25,
    min_points_for_kde: int = 30,
    figsize_per_panel: tuple = (4.5, 4.5),
) -> plt.Figure:
    """Small-multiples KDE contours: one subplot per viewpoint.

    Args:
        theta1_deg, theta2_deg: Angles in degrees [N].
        raw_vps: Fine-grained viewpoint labels [N].
        colors: VP -> color dict.
        kde_bw: Bandwidth for gaussian_kde.
        min_points_for_kde: Minimum class count for KDE contours.
        figsize_per_panel: Size of each subplot.

    Returns:
        Figure.
    """
    if colors is None:
        colors = VP_COLORS_8
    counts = Counter(raw_vps)
    vp_sorted = [v for v, _ in counts.most_common() if counts[v] >= 5]

    ncols = 4
    nrows = int(np.ceil(len(vp_sorted) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes).ravel()

    xgrid = np.linspace(-180, 180, 80)
    ygrid = np.linspace(-180, 180, 80)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    grid_pts = np.vstack([Xg.ravel(), Yg.ravel()])

    for ax_i, vp in enumerate(vp_sorted):
        ax = axes[ax_i]
        ax.scatter(theta1_deg, theta2_deg, c="lightgray", s=8, alpha=0.4, edgecolors="none")
        m = raw_vps == vp
        ax.scatter(
            theta1_deg[m], theta2_deg[m],
            c=colors.get(vp, "gray"), s=25, alpha=0.75, edgecolors="none",
        )
        if m.sum() >= min_points_for_kde:
            xy = np.vstack([theta1_deg[m], theta2_deg[m]])
            kde = gaussian_kde(xy, bw_method=kde_bw)
            Z = kde(grid_pts).reshape(Xg.shape)
            ax.contour(Xg, Yg, Z, levels=5, colors="black", linewidths=1.0, alpha=0.7)
        ax.set_title(f"{vp} (n={m.sum()})", fontsize=11)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.grid(True, alpha=0.3)

    for k in range(len(vp_sorted), len(axes)):
        axes[k].axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Angle 1 [°]")
    for ax in axes[::ncols]:
        ax.set_ylabel("Angle 2 [°]")

    plt.suptitle("Per-viewpoint distribution with KDE contours", fontsize=14, y=1.005)
    plt.tight_layout()
    return fig


def plot_torus_tiled(
    theta1_deg: np.ndarray,
    theta2_deg: np.ndarray,
    raw_vps: np.ndarray,
    colors: Dict[str, str] | None = None,
    figsize: tuple = (13, 13),
) -> plt.Figure:
    """Tile the (angle1, angle2) plane 3x3 to show toroidal periodicity.

    Args:
        theta1_deg, theta2_deg: Angles in degrees [N].
        raw_vps: Fine-grained viewpoint labels [N].
        colors: VP -> color dict.
        figsize: Figure size.

    Returns:
        Figure.
    """
    if colors is None:
        colors = VP_COLORS_8
    counts = Counter(raw_vps)
    vp_sorted = [v for v, _ in counts.most_common() if counts[v] >= 5]

    fig, ax = plt.subplots(figsize=figsize)

    for dx in [-360, 0, 360]:
        for dy in [-360, 0, 360]:
            is_center = (dx == 0 and dy == 0)
            alpha = 0.65 if is_center else 0.15
            size = 22 if is_center else 10
            for vp in vp_sorted:
                m = raw_vps == vp
                ax.scatter(
                    theta1_deg[m] + dx, theta2_deg[m] + dy,
                    c=colors.get(vp, "gray"), s=size, alpha=alpha,
                    edgecolors="none",
                )

    for k in range(4):
        x0 = -540 + k * 360
        ax.axvline(x0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(x0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    rect = mpatches.Rectangle((-180, -180), 360, 360, linewidth=2.5,
                              edgecolor="black", facecolor="none")
    ax.add_patch(rect)

    handles = [
        plt.Line2D([], [], marker="o", color=colors.get(v, "gray"), linestyle="",
                   markersize=10, label=v)
        for v in vp_sorted
    ]
    ax.legend(handles=handles, fontsize=10, ncol=2, loc="upper left",
              bbox_to_anchor=(1.02, 1))

    ax.set_xlabel("Angle 1: atan2(EV2, EV1) [degrees]")
    ax.set_ylabel("Angle 2: atan2(EV3, EV2) [degrees]")
    ax.set_title("Toroidal view — fundamental domain tiled 3x3\n(dashed lines = ±180° wrap)")
    ax.set_xlim(-540, 540)
    ax.set_ylim(-540, 540)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig
