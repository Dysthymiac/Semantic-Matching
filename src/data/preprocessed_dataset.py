"""Dataset class for managing preprocessed detections with efficient batch storage."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .coco_loader import COCOAnnotation
from ..utils.batch_storage import (
    save_batch_to_file,
    load_batch_from_file,
    get_batch_file_size_mb,
    analyze_batch_content,
    clean_batch_from_index
)


@dataclass(frozen=True)
class Detection:
    """Single detection with spatial features."""
    detection_id: str  # Unique ID for this detection
    image_path: str    # Path to source image
    species: str       # Species label
    bbox: torch.Tensor  # Bounding box: [4] (x1, y1, x2, y2)
    square_crop_bbox: torch.Tensor  # Square crop bbox: [4] (x1, y1, x2, y2) - may extend outside image bounds
    confidence_score: float  # Detection confidence
    features: torch.Tensor  # Spatial features: [embed_dim, H_patches, W_patches]
    patch_mask: torch.Tensor  # Patch-level mask: [H_patches, W_patches] - boolean mask for valid patches


class PreprocessedDataset:
    """Manages preprocessed detections with efficient batch storage."""

    def __init__(self, output_root: Path, batch_size: int = 1000) -> None:
        self.output_root = Path(output_root)
        self.batch_size = batch_size
        self.batches_dir = self.output_root / "batches"
        self.index_file = self.output_root / "index.pkl"

        # Create directory if it doesn't exist
        self.batches_dir.mkdir(parents=True, exist_ok=True)

        self._load_index()
        self._last_corrupted_images = set()  # Track images from last corruption

    def get_and_clear_corrupted_images(self) -> set:
        """Get images that were corrupted in last operation and clear the list."""
        corrupted = self._last_corrupted_images
        self._last_corrupted_images = set()
        return corrupted

    def _load_index(self) -> None:
        """Load or initialize the index."""
        if self.index_file.exists():
            with open(self.index_file, 'rb') as f:
                self._index = pickle.load(f)
        else:
            self._index = {
                'detection_to_batch': {},  # detection_id -> batch_file_path
                'batch_to_detections': {},  # batch_file_path -> list of detection_ids
                'image_to_detections': {},  # image_path -> list of detection_ids
                'processed_images': set(),  # Set of processed image paths
                'current_batch_id': 0,
                'current_batch_count': 0
            }

    def _save_index(self) -> None:
        """Save the index to disk."""
        with open(self.index_file, 'wb') as f:
            pickle.dump(self._index, f)

    def _get_current_batch_path(self) -> Path:
        """Get path for current batch file."""
        return self.batches_dir / f"batch_{self._index['current_batch_id']:06d}.pt"

    def _create_new_batch(self) -> None:
        """Start a new batch."""
        self._index['current_batch_id'] += 1
        self._index['current_batch_count'] = 0

    def add_detections_for_image(self, detections: List[Detection]) -> None:
        """Add all detections for a single image to the dataset."""
        if not detections:
            return

        image_path = detections[0].image_path

        # Check if we need a new batch (consider image as atomic unit)
        if self._index['current_batch_count'] > 0 and (self._index['current_batch_count'] + len(detections) > self.batch_size):
            self._create_new_batch()

        batch_path = self._get_current_batch_path()

        # Load existing batch or create new one
        batch_data = load_batch_from_file(batch_path)

        if not batch_data:
            # File doesn't exist, is empty, or was corrupted
            batch_rel_path = str(batch_path.relative_to(self.output_root))
            if batch_path.exists():
                # File existed but was corrupted - clean up index
                print(f"Corrupted batch file {batch_path}, recreating")
                corrupted_images = clean_batch_from_index(batch_rel_path, self._index)
                self._last_corrupted_images = corrupted_images
            else:
                # New batch file
                self._index['batch_to_detections'][batch_rel_path] = []

        # Add all detections to batch
        batch_rel_path = str(batch_path.relative_to(self.output_root))
        detection_ids = []

        for detection in detections:
            batch_data[detection.detection_id] = detection
            self._index['detection_to_batch'][detection.detection_id] = batch_rel_path
            self._index['batch_to_detections'][batch_rel_path].append(detection.detection_id)
            detection_ids.append(detection.detection_id)

        # Update counters and mappings
        self._index['current_batch_count'] += len(detections)
        self._index['processed_images'].add(image_path)
        self._index['image_to_detections'][image_path] = detection_ids

        # Save updated batch using optimized storage
        save_batch_to_file(batch_data, batch_path)

        # Debug: Check actual batch file size and detection content
        batch_file_size_mb = get_batch_file_size_mb(batch_path)
        print(f"DEBUG: Saved batch with {len(batch_data)} detections, file size: {batch_file_size_mb:.1f}MB")

        if len(batch_data) > 0:
            avg_size_mb = batch_file_size_mb / len(batch_data)
            print(f"DEBUG: Average {avg_size_mb:.1f}MB per detection")

            # Analyze batch content using utility function
            content_stats = analyze_batch_content(batch_data)
            print(f"DEBUG: Features: {content_stats['features_size_mb']:.1f}MB {content_stats['features_shape']}, "
                  f"Mask: {content_stats['patch_mask_size_kb']:.1f}KB, "
                  f"dtype: {content_stats['features_dtype']}")

        # Save index
        self._save_index()

    def add_detections_for_batch(self, all_detections: List[Detection]) -> None:
        """Add all detections from a processing batch efficiently - single load/save operation."""
        if not all_detections:
            return

        # Group detections by image path to update index correctly
        detections_by_image = {}
        for detection in all_detections:
            image_path = detection.image_path
            if image_path not in detections_by_image:
                detections_by_image[image_path] = []
            detections_by_image[image_path].append(detection)

        # Check if we need a new batch file for all these detections
        total_new_detections = len(all_detections)
        if self._index['current_batch_count'] > 0 and (self._index['current_batch_count'] + total_new_detections > self.batch_size):
            self._create_new_batch()

        batch_path = self._get_current_batch_path()

        # Load existing batch ONCE
        batch_data = load_batch_from_file(batch_path)
        if not batch_data:
            batch_rel_path = str(batch_path.relative_to(self.output_root))
            if batch_path.exists():
                # File existed but was corrupted - clean up index
                print(f"Corrupted batch file {batch_path}, recreating")
                corrupted_images = clean_batch_from_index(batch_rel_path, self._index)
                self._last_corrupted_images = corrupted_images
            else:
                # New batch file
                self._index['batch_to_detections'][batch_rel_path] = []

        # Add ALL detections to batch data
        batch_rel_path = str(batch_path.relative_to(self.output_root))

        for image_path, image_detections in detections_by_image.items():
            detection_ids = []

            for detection in image_detections:
                batch_data[detection.detection_id] = detection
                self._index['detection_to_batch'][detection.detection_id] = batch_rel_path
                self._index['batch_to_detections'][batch_rel_path].append(detection.detection_id)
                detection_ids.append(detection.detection_id)

            # Update image tracking
            self._index['processed_images'].add(image_path)
            self._index['image_to_detections'][image_path] = detection_ids

        # Update batch count
        self._index['current_batch_count'] += total_new_detections

        # Save batch ONCE with all detections
        save_batch_to_file(batch_data, batch_path)

        # Debug info
        batch_file_size_mb = get_batch_file_size_mb(batch_path)
        print(f"DEBUG: Saved batch with {len(batch_data)} detections, file size: {batch_file_size_mb:.1f}MB")

        if len(batch_data) > 0:
            avg_size_mb = batch_file_size_mb / len(batch_data)
            print(f"DEBUG: Average {avg_size_mb:.1f}MB per detection")

            # Analyze batch content
            content_stats = analyze_batch_content(batch_data)
            print(f"DEBUG: Features: {content_stats['features_size_mb']:.1f}MB {content_stats['features_shape']}, "
                  f"Mask: {content_stats['patch_mask_size_kb']:.1f}KB, "
                  f"dtype: {content_stats['features_dtype']}")

        # Save index
        self._save_index()

    def get_detection(self, detection_id: str) -> Optional[Detection]:
        """Get a single detection by ID."""
        if detection_id not in self._index['detection_to_batch']:
            return None

        batch_path = self.output_root / self._index['detection_to_batch'][detection_id]
        batch_data = load_batch_from_file(batch_path)

        return batch_data.get(detection_id)

    def get_detections_for_image(self, image_path: str) -> List[Detection]:
        """Get all detections for a specific image efficiently."""
        if image_path not in self._index['image_to_detections']:
            return []

        detection_ids = self._index['image_to_detections'][image_path]
        detections = []

        for detection_id in detection_ids:
            detection = self.get_detection(detection_id)
            if detection:
                detections.append(detection)

        return detections

    def iter_all_detections(self) -> List[Detection]:
        """Iterate through all detections efficiently by loading batches."""
        all_detections = []

        for batch_rel_path in self._index['batch_to_detections'].keys():
            batch_path = self.output_root / batch_rel_path
            if batch_path.exists():
                batch_data = load_batch_from_file(batch_path)
                all_detections.extend(batch_data.values())

        return all_detections

    def is_image_processed(self, image_path: str) -> bool:
        """Check if image has been processed."""
        return image_path in self._index['processed_images']

    def get_unprocessed_images(self, all_image_paths: List[str]) -> List[str]:
        """Filter image paths to only unprocessed ones."""
        return [path for path in all_image_paths if not self.is_image_processed(path)]

    def get_total_detection_count(self) -> int:
        """Get total number of detections."""
        return len(self._index['detection_to_batch'])

    def get_batch_count(self) -> int:
        """Get number of batch files."""
        return len(self._index['batch_to_detections'])

    def get_processed_image_count(self) -> int:
        """Get count of processed images."""
        return len(self._index['processed_images'])