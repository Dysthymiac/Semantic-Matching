"""Graph Laplacian spectral methods for manifold learning."""

from .graph import build_knn_graph, normalized_laplacian
from .spectral import smallest_eigenvectors
from .angles import angles_from_evs, angle_pair
from .diagnostics import ev_health_check, find_bottleneck_evs

__all__ = [
    "build_knn_graph",
    "normalized_laplacian",
    "smallest_eigenvectors",
    "angles_from_evs",
    "angle_pair",
    "ev_health_check",
    "find_bottleneck_evs",
]
