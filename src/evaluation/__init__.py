"""Evaluation utilities for detection matching and re-identification metrics."""

from .detection_matching import (
    match_detections_to_gt,
    match_detections_to_gt_patch_overlap,
    compute_iou,
    get_identity_mapping,
    get_image_uuid_from_detection_id,
    save_matching,
    load_matching,
    load_or_compute_matching,
)
from .reid_metrics import compute_reid_accuracy
from .viewpoint_accuracy import (
    COARSE_MAP,
    coarse_viewpoint,
    relaxed_correct,
    relaxed_same_viewpoint,
)
from .discriminability import fisher_discriminant_1d, pairwise_same_class_auc
from .texture_diagnostic import (
    TextureDiagnosticResult,
    run_texture_diagnostic,
    compute_texture_identity_separation,
    load_detection_features,
    plot_diagnostic_distributions,
)

__all__ = [
    "match_detections_to_gt",
    "match_detections_to_gt_patch_overlap",
    "compute_iou",
    "get_identity_mapping",
    "get_image_uuid_from_detection_id",
    "save_matching",
    "load_matching",
    "load_or_compute_matching",
    "compute_reid_accuracy",
    "TextureDiagnosticResult",
    "run_texture_diagnostic",
    "compute_texture_identity_separation",
    "load_detection_features",
    "plot_diagnostic_distributions",
    "COARSE_MAP",
    "coarse_viewpoint",
    "relaxed_correct",
    "relaxed_same_viewpoint",
    "fisher_discriminant_1d",
    "pairwise_same_class_auc",
]
