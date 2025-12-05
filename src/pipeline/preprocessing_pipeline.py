"""Complete preprocessing pipeline combining SAM3 segmentation and DINOv3 feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch

from ..config.config import MainConfig
from ..segmentation.sam3_segmenter import SAM3Segmenter
from ..features.dino_extractor import DINOv3Extractor
from ..data.preprocessed_dataset import Detection, PreprocessedDataset
from ..utils.crop_utils import extract_cropped_detections


def generate_detection_id(image_path: str, detection_idx: int) -> str:
    """Generate unique detection ID."""
    image_name = Path(image_path).stem
    return f"{image_name}_det_{detection_idx:04d}"


def collect_all_detections_for_dino(images: List[Image.Image], image_paths: List[str],
                                   segmentation_outputs: List[dict],
                                   target_size: int) -> Tuple[List, List, List]:
    """
    Collect all cropped detections from all images for batch DINO processing.

    Returns:
        cropped_images: List of all cropped detection images
        cropped_masks: List of all cropped detection masks
        detection_metadata: List of (img_idx, detection_idx, bbox, square_crop_bbox, score) tuples
    """
    all_cropped_images = []
    all_cropped_masks = []
    detection_metadata = []

    for img_idx, (image, seg_output) in enumerate(zip(images, segmentation_outputs)):
        if len(seg_output["masks"]) > 0:
            cropped_detections = extract_cropped_detections(image, seg_output, target_size)

            if not cropped_detections:
                print(f"Failed to crop detections for: {image_paths[img_idx]}")
                continue

            for detection_idx, (cropped_img, cropped_mask, bbox, square_crop_bbox, _, _) in enumerate(cropped_detections):
                all_cropped_images.append(cropped_img)
                all_cropped_masks.append(cropped_mask)
                detection_metadata.append((img_idx, detection_idx, bbox, square_crop_bbox, seg_output["scores"][detection_idx]))

    return all_cropped_images, all_cropped_masks, detection_metadata


def process_detections_with_dino_batching(all_cropped_images: List, all_cropped_masks: List,
                                         detection_metadata: List[Tuple], dino_extractor,
                                         max_batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process large sets of detections with DINOv3 using sub-batching to avoid OOM.

    Single responsibility: Handle DINOv3 batching for memory management.
    """
    if not all_cropped_images:
        return torch.empty((0, 1024, 32, 32)), torch.empty((0, 1, 32, 32))

    total_detections = len(all_cropped_images)
    print(f"DINOv3 processing {total_detections} detections in batches of {max_batch_size}")

    all_features = []
    all_masks = []

    # Process detections in sub-batches
    for start_idx in range(0, total_detections, max_batch_size):
        end_idx = min(start_idx + max_batch_size, total_detections)

        batch_images = all_cropped_images[start_idx:end_idx]
        batch_masks = all_cropped_masks[start_idx:end_idx]

        print(f"  DINOv3 sub-batch: {len(batch_images)} detections ({start_idx}-{end_idx-1})")

        # Process this sub-batch
        features_batch, masks_batch = dino_extractor.extract_features_with_masks(batch_images, batch_masks)

        all_features.append(features_batch)
        all_masks.append(masks_batch)

        # Clear GPU cache between sub-batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all results
    features_spatial = torch.cat(all_features, dim=0)
    masks_spatial = torch.cat(all_masks, dim=0)

    print(f"DINOv3 completed: {features_spatial.shape[0]} features processed")
    return features_spatial, masks_spatial


def create_detections_from_features(detection_metadata: List[Tuple], features_spatial, masks_spatial,
                                   image_paths: List[str], species: str) -> List[Tuple[int, Detection]]:
    """
    Create Detection objects from DINO features and metadata.

    Returns:
        List of (img_idx, Detection) tuples for organizing by image
    """
    detections_with_image_idx = []
    detection_counters = {}

    for feature_idx, (img_idx, detection_idx, bbox, square_crop_bbox, score) in enumerate(detection_metadata):
        if feature_idx < features_spatial.shape[0]:
            # Track detection counter per image
            if img_idx not in detection_counters:
                detection_counters[img_idx] = 0

            # Get spatial features and mask for this detection
            detection_features = features_spatial[feature_idx]  # [embed_dim, H_patches, W_patches]
            detection_patch_mask = masks_spatial[feature_idx]  # [1, H_patches, W_patches]

            # Ensure all tensors are on CPU
            detection_features_cpu = detection_features.cpu()
            detection_patch_mask_cpu = detection_patch_mask.squeeze(0).cpu()
            bbox_cpu = bbox.cpu()
            square_crop_bbox_cpu = square_crop_bbox.cpu()

            detection = Detection(
                detection_id=generate_detection_id(image_paths[img_idx], detection_counters[img_idx]),
                image_path=str(image_paths[img_idx]),
                species=species,
                bbox=bbox_cpu,
                square_crop_bbox=square_crop_bbox_cpu,
                confidence_score=float(score),
                features=detection_features_cpu,
                patch_mask=detection_patch_mask_cpu
            )
            detections_with_image_idx.append((img_idx, detection))
            detection_counters[img_idx] += 1

    return detections_with_image_idx


class PreprocessingPipeline:
    """Complete preprocessing pipeline for SAM3 + DINOv3 processing."""

    def __init__(self, config: MainConfig):
        self.config = config
        self.sam_segmenter = SAM3Segmenter(config.sam)
        self.dino_extractor = DINOv3Extractor(config.dino)

    def process_images_batch(self, images: List[Image.Image], image_paths: List[str], species_list: List[str]) -> List[List[Detection]]:
        """Process a batch of images with SAM3 segmentation and DINOv3 feature extraction."""
        if not images or len(images) != len(image_paths):
            return []

        print(f"Processing batch of {len(images)} images")

        # Initialize results - list of detections for each image
        batch_results = [[] for _ in range(len(images))]

        for species in species_list:
            # Step 1: SAM3 segmentation for all images
            species_batch = [species] * len(images)
            segmentation_outputs = self.sam_segmenter.segment_images_batch(images, species_batch)

            # Filter by confidence for each image
            filtered_outputs = []
            min_score = self.config.sam.min_score
            for img_idx, output in enumerate(segmentation_outputs):
                filtered = self.sam_segmenter.filter_masks_by_score(output, min_score=min_score)

                # Only print if there are issues
                if len(output['masks']) == 0:
                    print(f"SAM3 found no detections for: {image_paths[img_idx]}")
                elif len(filtered['masks']) == 0:
                    print(f"All {len(output['masks'])} detections filtered out (below {min_score} confidence) for: {image_paths[img_idx]}")

                filtered_outputs.append(filtered)

            # Step 2: Collect ALL cropped detections from ALL images
            all_cropped_images, all_cropped_masks, detection_metadata = collect_all_detections_for_dino(
                images, image_paths, filtered_outputs, self.config.dino.resize_size
            )

            if not all_cropped_images:
                continue  # No detections found for this species

            # Step 3: Batch process ALL detections with DINOv3 using sub-batching
            features_spatial, masks_spatial = process_detections_with_dino_batching(
                all_cropped_images, all_cropped_masks, detection_metadata,
                self.dino_extractor, max_batch_size=self.config.dino.batch_size
            )

            # Check for silent DINOv3 failures
            if features_spatial.numel() == 0:
                print(f"DINOv3 produced no features for species: {species}")
                continue

            # Step 4: Create Detection objects and organize by image
            detections_with_image_idx = create_detections_from_features(
                detection_metadata, features_spatial, masks_spatial, image_paths, species
            )

            # Organize detections back into per-image results
            for img_idx, detection in detections_with_image_idx:
                batch_results[img_idx].append(detection)

        return batch_results

    def process_single_image(self, image: Image.Image, image_path: str, species_list: List[str]) -> List[Detection]:
        """Process a single image - wrapper for batch method."""
        results = self.process_images_batch([image], [image_path], species_list)
        return results[0] if results else []