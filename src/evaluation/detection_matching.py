"""Match predicted detections to ground truth annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..data.coco_loader import COCOLoader, COCOAnnotation, BoundingBox
from ..data.preprocessed_dataset import PreprocessedDataset
from ..utils.memory_monitor import force_garbage_collection


@dataclass
class MatchedDetection:
    """A predicted detection matched to a ground truth annotation.

    Note: Stores only detection_id (not full Detection) to avoid OOM
    when matching thousands of detections with large feature tensors.
    """
    detection_id: str
    gt_annotation: COCOAnnotation
    iou: float


def get_image_uuid_from_detection_id(detection_id: str) -> str:
    """Extract image UUID from detection_id (format: {image_uuid}_det_{index})."""
    return detection_id.rsplit("_det_", 1)[0]


def compute_iou(box1: torch.Tensor, box2: BoundingBox) -> float:
    """Compute IoU between a detection bbox tensor [x1,y1,x2,y2] and a GT BoundingBox."""
    x1_1, y1_1, x2_1, y2_1 = box1[0].item(), box1[1].item(), box1[2].item(), box1[3].item()
    x1_2, y1_2, x2_2, y2_2 = box2.x1, box2.y1, box2.x2, box2.y2

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def match_detections_to_gt(
    dataset: PreprocessedDataset,
    coco_loader: COCOLoader,
    iou_threshold: float = 0.5,
    category_ids: Optional[List[int]] = None,
) -> List[MatchedDetection]:
    """
    Match predicted detections to ground truth annotations by IoU.

    Args:
        dataset: Preprocessed detection dataset
        coco_loader: COCO annotations loader
        iou_threshold: Minimum IoU for a match
        category_ids: Filter GT annotations by category (e.g., [1] for zebra_grevys)

    Returns:
        List of matched detections with their GT annotations
    """
    # Build image_uuid -> GT annotations mapping
    gt_by_image: Dict[str, List[COCOAnnotation]] = {}
    for ann in coco_loader.annotations:
        # Filter by category if specified
        if category_ids and ann.category_id not in category_ids:
            continue
        if ann.image_uuid not in gt_by_image:
            gt_by_image[ann.image_uuid] = []
        gt_by_image[ann.image_uuid].append(ann)

    matched = []

    # Iterate over batches efficiently instead of loading each detection individually
    batch_to_detections = dataset._index.get('batch_to_detections', {})

    from ..utils.batch_storage import load_batch_from_file

    for batch_rel_path in tqdm(batch_to_detections.keys(), desc="Matching detections (by batch)"):
        batch_path = dataset.output_root / batch_rel_path
        if not batch_path.exists():
            continue

        batch_data = load_batch_from_file(batch_path)
        if not batch_data:
            continue

        for det_id, det in batch_data.items():
            # Extract image UUID from detection_id (format: {image_uuid}_det_{index})
            image_uuid = get_image_uuid_from_detection_id(det_id)

            # Get GT annotations for this image
            gt_annotations = gt_by_image.get(image_uuid, [])
            if not gt_annotations:
                continue

            # Find best matching GT annotation by IoU
            best_iou = 0.0
            best_gt = None

            for gt_ann in gt_annotations:
                iou = compute_iou(det.bbox, gt_ann.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_ann

            if best_iou >= iou_threshold and best_gt is not None:
                matched.append(MatchedDetection(
                    detection_id=det_id,
                    gt_annotation=best_gt,
                    iou=best_iou
                ))

        # Clean up batch data after processing
        del batch_data
        force_garbage_collection()

    return matched


def get_identity_mapping(matched_detections: List[MatchedDetection]) -> Dict[str, str]:
    """
    Create mapping from detection_id to individual_id.

    Args:
        matched_detections: List of matched detections

    Returns:
        Dict mapping detection_id -> individual_id
    """
    return {
        m.detection_id: m.gt_annotation.individual_id
        for m in matched_detections
    }
