"""Feature extraction modules."""

from .dino_extractor import (
    DINOv3Extractor,
    resize_and_pad_image,
    get_valid_patch_mask,
)
from .dedode_extractor import (
    DeDoDeExtractor,
    make_patch_grid_keypoints,
)
from .roma_extractor import RoMaExtractor
from .sift_extractor import SIFTExtractor
from .disk_extractor import DISKExtractor
from .joint_fisher_vector import (
    encode_mlr_fisher_vector,
    encode_detection_mlr_fisher_vector,
    compute_mlr_posteriors,
    compute_semantic_posteriors,
    get_mlr_fv_dimension,
)

__all__ = [
    "DINOv3Extractor",
    "resize_and_pad_image",
    "get_valid_patch_mask",
    "DeDoDeExtractor",
    "make_patch_grid_keypoints",
    "RoMaExtractor",
    "SIFTExtractor",
    "DISKExtractor",
    "encode_mlr_fisher_vector",
    "encode_detection_mlr_fisher_vector",
    "compute_mlr_posteriors",
    "compute_semantic_posteriors",
    "get_mlr_fv_dimension",
]