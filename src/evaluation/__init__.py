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
]
