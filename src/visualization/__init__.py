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
from .patch_matches import (
    draw_patch_matches,
    visualize_patch_matches,
    patch_grid_to_crop_center,
)
from .mlr_vis import (
    compute_mlr_patch_responsibilities,
    compute_semantic_patch_responsibilities,
)
from .residual_fisher_vis import (
    ResidualFisherParams,
    load_residual_fisher_model,
    compute_residual_fisher_joint_responsibilities,
    compute_residual_fisher_semantic_responsibilities,
    compute_posterior_shift,
)
from .fisher_similarity_vis import (
    SimilarityDecomposition,
    decompose_fisher_similarity,
    compute_patch_contribution_map,
    get_top_contributing_components,
    visualize_similarity_explanation,
    visualize_component_spatial_distribution,
)
