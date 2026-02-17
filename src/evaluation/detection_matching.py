"""Match predicted detections to ground truth annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from ..data.coco_loader import COCOLoader, COCOAnnotation, BoundingBox
from ..data.preprocessed_dataset import PreprocessedDataset
from ..utils.memory_monitor import force_garbage_collection
from ..visualization.primitives import get_crop_bounds, compute_padding_info


@dataclass
class MatchedDetection:
    """A predicted detection matched to a ground truth annotation.

    Note: Stores only detection_id (not full Detection) to avoid OOM
    when matching thousands of detections with large feature tensors.
    """
    detection_id: str
    gt_annotation: COCOAnnotation
    score: float


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


def patch_centers_in_image_space(
    square_crop_bbox: torch.Tensor,
    patch_mask: torch.Tensor,
    img_w: int,
    img_h: int,
    target_size: int,
    patch_size: int,
) -> List[Tuple[float, float]]:
    """Map valid patch centers to image-space coordinates.

    Args:
        square_crop_bbox: [4] tensor (x1, y1, x2, y2) — may extend outside image
        patch_mask: [H_patches, W_patches] boolean mask
        img_w, img_h: Original image dimensions
        target_size: Image resize target (e.g. 512)
        patch_size: Patch size in pixels (e.g. 16)

    Returns:
        List of (img_x, img_y) for each valid patch.
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = get_crop_bounds(square_crop_bbox, img_w, img_h)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w <= 0 or crop_h <= 0:
        return []

    scale, pad_left, pad_top = compute_padding_info(crop_w, crop_h, target_size)

    half_patch = patch_size // 2
    centers = []
    h_patches, w_patches = patch_mask.shape

    for i in range(h_patches):
        for j in range(w_patches):
            if patch_mask[i, j]:
                px = j * patch_size + half_patch
                py = i * patch_size + half_patch
                img_x = crop_x1 + (px - pad_left) / scale
                img_y = crop_y1 + (py - pad_top) / scale
                centers.append((img_x, img_y))

    return centers


def _point_in_bbox(x: float, y: float, bbox: BoundingBox) -> bool:
    """Check if a point falls within a bounding box."""
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2


def _bboxes_overlap(det_bbox: torch.Tensor, gt_bbox: BoundingBox) -> bool:
    """Quick check if detection bbox overlaps GT bbox at all."""
    x1_1, y1_1, x2_1, y2_1 = det_bbox[0].item(), det_bbox[1].item(), det_bbox[2].item(), det_bbox[3].item()
    return not (x2_1 <= gt_bbox.x1 or x1_1 >= gt_bbox.x2 or
                y2_1 <= gt_bbox.y1 or y1_1 >= gt_bbox.y2)


def match_detections_to_gt_patch_overlap(
    dataset: PreprocessedDataset,
    coco_loader: COCOLoader,
    target_size: int,
    patch_size: int,
    category_ids: Optional[List[int]] = None,
    min_overlap_fraction: float = 0.2,
) -> List[MatchedDetection]:
    """Match detections to GT annotations using patch-position overlap.

    For each GT annotation, finds the detection whose valid patches
    overlap the GT bbox the most. Uses strict 1-to-1 greedy assignment.

    Args:
        dataset: Preprocessed detection dataset
        coco_loader: COCO annotations loader
        target_size: Image resize target (from config)
        patch_size: Patch size in pixels (from config)
        category_ids: Filter GT annotations by category
        min_overlap_fraction: Minimum fraction of a detection's patches
            that must fall inside the GT bbox to be considered a candidate.
            Filters out accidental overlaps.

    Returns:
        List of matched detections with their GT annotations.
    """
    # Build image_uuid -> GT annotations mapping
    gt_by_image: Dict[str, List[COCOAnnotation]] = {}
    for ann in coco_loader.annotations:
        if category_ids and ann.category_id not in category_ids:
            continue
        # Skip region-only annotations — their bbox is the identifiable region,
        # not the full animal, so matching against full detections is wrong.
        if ann.annot_census_region:
            continue
        gt_by_image.setdefault(ann.image_uuid, []).append(ann)

    # Build image_uuid -> list of (det_id, Detection) for all detections in those images
    # We need patch_mask and square_crop_bbox, so we must load batch data
    from ..utils.batch_storage import load_batch_from_file

    batch_to_detections = dataset._index.get('batch_to_detections', {})

    # Collect all (gt_ann, det_id, n_patches_inside) triples
    triples: List[Tuple[COCOAnnotation, str, int]] = []

    for batch_rel_path in tqdm(batch_to_detections.keys(), desc="Computing patch overlaps"):
        batch_path = dataset.output_root / batch_rel_path
        if not batch_path.exists():
            continue

        batch_data = load_batch_from_file(batch_path)
        if not batch_data:
            continue

        for det_id, det in batch_data.items():
            image_uuid = get_image_uuid_from_detection_id(det_id)
            gt_annotations = gt_by_image.get(image_uuid, [])
            if not gt_annotations:
                continue

            # Get image dimensions
            coco_image = coco_loader._images.get(image_uuid)
            if coco_image is None:
                continue

            # Compute patch centers in image space (once per detection)
            centers = patch_centers_in_image_space(
                det.square_crop_bbox, det.patch_mask,
                coco_image.width, coco_image.height,
                target_size, patch_size,
            )

            if not centers:
                continue

            # Score this detection against each overlapping GT
            for gt_ann in gt_annotations:
                if not _bboxes_overlap(det.bbox, gt_ann.bbox):
                    continue

                n_inside = sum(1 for x, y in centers if _point_in_bbox(x, y, gt_ann.bbox))
                fraction = n_inside / len(centers)

                if fraction >= min_overlap_fraction:
                    triples.append((gt_ann, det_id, n_inside))

        del batch_data
        force_garbage_collection()

    # Strict 1-to-1 greedy assignment: sort by score descending
    triples.sort(key=lambda t: t[2], reverse=True)

    used_gts: set = set()
    used_dets: set = set()
    matched: List[MatchedDetection] = []

    for gt_ann, det_id, overlap_score in triples:
        if gt_ann.uuid in used_gts or det_id in used_dets:
            continue
        matched.append(MatchedDetection(
            detection_id=det_id,
            gt_annotation=gt_ann,
            score=overlap_score,
        ))
        used_gts.add(gt_ann.uuid)
        used_dets.add(det_id)

    print(f"Matched {len(matched)} detections to GT (from {sum(len(v) for v in gt_by_image.values())} GT annotations)")

    return matched


# --- Legacy matching (kept for backward compatibility) ---

def match_detections_to_gt(
    dataset: PreprocessedDataset,
    coco_loader: COCOLoader,
    iou_threshold: float = 0.5,
    category_ids: Optional[List[int]] = None,
) -> List[MatchedDetection]:
    """Match predicted detections to ground truth annotations by IoU.

    DEPRECATED: Use match_detections_to_gt_patch_overlap instead.
    This function uses per-detection greedy matching which can assign
    multiple detections to the same GT annotation.
    """
    gt_by_image: Dict[str, List[COCOAnnotation]] = {}
    for ann in coco_loader.annotations:
        if category_ids and ann.category_id not in category_ids:
            continue
        gt_by_image.setdefault(ann.image_uuid, []).append(ann)

    matched = []
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
            image_uuid = get_image_uuid_from_detection_id(det_id)
            gt_annotations = gt_by_image.get(image_uuid, [])
            if not gt_annotations:
                continue

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
                    score=best_iou,
                ))

        del batch_data
        force_garbage_collection()

    return matched


# --- Caching ---

def save_matching(
    matched: List[MatchedDetection],
    path: Path,
    params: dict,
) -> None:
    """Save matching results to JSON.

    Args:
        matched: List of matched detections
        path: Output file path
        params: Parameters used for matching (for cache validation)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": 2,
        "params": params,
        "matches": [
            {
                "detection_id": m.detection_id,
                "gt_uuid": m.gt_annotation.uuid,
                "gt_individual_id": m.gt_annotation.individual_id,
                "gt_viewpoint": m.gt_annotation.viewpoint,
                "gt_category_id": m.gt_annotation.category_id,
                "gt_image_uuid": m.gt_annotation.image_uuid,
                "gt_bbox": [m.gt_annotation.bbox.x1, m.gt_annotation.bbox.y1,
                            m.gt_annotation.bbox.x2, m.gt_annotation.bbox.y2],
                "score": m.score,
            }
            for m in matched
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(matched)} matching results to {path}")


def load_matching(
    path: Path,
) -> Optional[Tuple[List[MatchedDetection], dict]]:
    """Load matching results from JSON cache.

    Args:
        path: Cache file path

    Returns:
        (matched, params) tuple, or None if cache missing/invalid.
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    if data.get("version") != 2:
        return None

    params = data.get("params", {})
    matched = []

    for entry in data.get("matches", []):
        bbox_coords = entry.get("gt_bbox", [0, 0, 0, 0])
        gt_ann = COCOAnnotation(
            uuid=entry["gt_uuid"],
            image_uuid=entry.get("gt_image_uuid", ""),
            bbox=BoundingBox(*bbox_coords),
            individual_id=entry.get("gt_individual_id", ""),
            viewpoint=entry.get("gt_viewpoint", "unknown"),
            category_id=entry.get("gt_category_id", 0),
            annot_census=False,
        )
        matched.append(MatchedDetection(
            detection_id=entry["detection_id"],
            gt_annotation=gt_ann,
            score=entry.get("score", 0.0),
        ))

    print(f"Loaded {len(matched)} matching results from cache ({path})")
    return matched, params


def _resolve_category_names(
    coco_loader: COCOLoader,
    category_names: Optional[List[str]],
) -> Optional[List[int]]:
    """Resolve category names to IDs using COCOLoader.

    Args:
        coco_loader: COCO annotations loader (has _categories: {id: COCOCategory})
        category_names: List of category names (e.g. ["zebra_grevys"])

    Returns:
        List of category IDs, or None if no filtering requested.
    """
    if not category_names:
        return None

    name_to_id = {cat.species: cat_id for cat_id, cat in coco_loader._categories.items()}
    ids = []
    for name in category_names:
        if name in name_to_id:
            ids.append(name_to_id[name])
        else:
            print(f"WARNING: category '{name}' not found in COCO annotations. "
                  f"Available: {list(name_to_id.keys())}")

    return ids if ids else None


def load_or_compute_matching(
    dataset: PreprocessedDataset,
    coco_loader: COCOLoader,
    output_root: Path,
    target_size: int,
    patch_size: int,
    category_names: Optional[List[str]] = None,
    min_overlap_fraction: float = 0.2,
) -> List[MatchedDetection]:
    """Load matching from cache, or compute and save.

    Args:
        dataset: Preprocessed detection dataset
        coco_loader: COCO annotations loader
        output_root: Root directory for outputs
        target_size: Image resize target (from config)
        patch_size: Patch size in pixels (from config)
        category_names: Filter GT annotations by category name
            (e.g. ["zebra_grevys"]). None or empty means no filtering.
        min_overlap_fraction: Minimum fraction of a detection's patches
            inside GT bbox to be a candidate.

    Returns:
        List of matched detections.
    """
    category_ids = _resolve_category_names(coco_loader, category_names)

    cache_path = Path(output_root) / "matching" / "gt_matching.json"
    current_params = {
        "target_size": target_size,
        "patch_size": patch_size,
        "category_names": sorted(category_names) if category_names else None,
        "min_overlap_fraction": min_overlap_fraction,
    }

    result = load_matching(cache_path)
    if result is not None:
        matched, cached_params = result
        if cached_params == current_params:
            return matched
        print(f"Cache params mismatch, recomputing matching...")

    matched = match_detections_to_gt_patch_overlap(
        dataset, coco_loader, target_size, patch_size,
        category_ids=category_ids,
        min_overlap_fraction=min_overlap_fraction,
    )
    save_matching(matched, cache_path, current_params)
    return matched


# --- Utilities ---

def get_identity_mapping(matched_detections: List[MatchedDetection]) -> Dict[str, str]:
    """Create mapping from detection_id to individual_id.

    Args:
        matched_detections: List of matched detections

    Returns:
        Dict mapping detection_id -> individual_id
    """
    return {
        m.detection_id: m.gt_annotation.individual_id
        for m in matched_detections
    }
