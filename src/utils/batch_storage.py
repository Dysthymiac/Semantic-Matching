"""Standalone batch storage utilities for Detection objects."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Any

import torch


def save_batch_to_file(batch_data: Dict[str, Any], batch_path: Path) -> None:
    """
    Save batch data to file using torch.save for tensor optimization.

    Single responsibility: Only handles file saving with optimized format.
    """
    torch.save(batch_data, batch_path)


def load_batch_from_file(batch_path: Path) -> Dict[str, Any]:
    """
    Load batch data from file.

    Single responsibility: Only handles file loading.
    Fail-fast: Let torch.load crash immediately if file is corrupted.
    """
    if not batch_path.exists() or batch_path.stat().st_size == 0:
        return {}

    return torch.load(batch_path, weights_only=False)


def get_batch_file_size_mb(batch_path: Path) -> float:
    """
    Get batch file size in megabytes.

    Pure function: Simple file size calculation.
    """
    if not batch_path.exists():
        return 0.0
    return batch_path.stat().st_size / (1024 * 1024)


def analyze_batch_content(batch_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze batch content and return size statistics.

    Pure function: Only analyzes provided data without side effects.
    """
    if not batch_data:
        return {"detection_count": 0, "avg_size_mb": 0.0}

    first_detection = next(iter(batch_data.values()))
    features_size = first_detection.features.numel() * first_detection.features.element_size()
    patch_mask_size = first_detection.patch_mask.numel() * first_detection.patch_mask.element_size()

    return {
        "detection_count": len(batch_data),
        "features_shape": tuple(first_detection.features.shape),
        "features_dtype": str(first_detection.features.dtype),
        "features_size_mb": features_size / (1024 * 1024),
        "patch_mask_size_kb": patch_mask_size / 1024
    }


def extract_images_from_batch_index(batch_rel_path: str, index: Dict[str, Any]) -> Set[str]:
    """
    Extract all image paths that have detections in the specified batch.

    Pure function: Only processes index data to find affected images.
    """
    if batch_rel_path not in index.get('batch_to_detections', {}):
        return set()

    detection_ids = index['batch_to_detections'][batch_rel_path]
    affected_images = set()

    for image_path, img_detection_ids in index.get('image_to_detections', {}).items():
        for detection_id in detection_ids:
            if detection_id in img_detection_ids:
                affected_images.add(image_path)
                break

    return affected_images


def clean_batch_from_index(batch_rel_path: str, index: Dict[str, Any]) -> Set[str]:
    """
    Clean all references to a corrupted batch from index and return affected images.

    Single responsibility: Only handles index cleanup for one batch.
    """
    affected_images = extract_images_from_batch_index(batch_rel_path, index)

    if batch_rel_path not in index.get('batch_to_detections', {}):
        return affected_images

    # Get detection IDs that were in this batch
    detection_ids = index['batch_to_detections'][batch_rel_path]

    # Remove affected images from processed set
    for image_path in affected_images:
        if image_path in index.get('processed_images', set()):
            index['processed_images'].remove(image_path)
        # Remove image's detections from index
        if image_path in index.get('image_to_detections', {}):
            del index['image_to_detections'][image_path]

    # Clean up detection mappings
    for detection_id in detection_ids:
        if detection_id in index.get('detection_to_batch', {}):
            del index['detection_to_batch'][detection_id]

    # Reset batch mapping
    index['batch_to_detections'][batch_rel_path] = []

    return affected_images