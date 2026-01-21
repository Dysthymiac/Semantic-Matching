"""Evaluation utilities for detection matching and re-identification metrics."""

from .detection_matching import (
    match_detections_to_gt,
    compute_iou,
    get_identity_mapping,
    get_image_uuid_from_detection_id,
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
    "compute_iou",
    "get_identity_mapping",
    "get_image_uuid_from_detection_id",
    "compute_reid_accuracy",
    "TextureDiagnosticResult",
    "run_texture_diagnostic",
    "compute_texture_identity_separation",
    "load_detection_features",
    "plot_diagnostic_distributions",
]
