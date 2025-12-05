"""Incremental PCA utilities for large-scale feature preprocessing."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

from ..config.config import PCAConfig
from ..data.preprocessed_dataset import Detection


class IncrementalPCAProcessor:
    """Handles incremental PCA fitting and transformation of DINO features."""

    def __init__(self, config: PCAConfig, output_root: Path) -> None:
        self.config = config
        self.output_root = Path(output_root)
        self.pca_dir = self.output_root / "pca"
        self.pca_state_path = self.pca_dir / "pca_state.pkl"

        # Create PCA directory
        self.pca_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load PCA model
        self.pca = self._create_pca_model()
        self.stats = self._load_or_create_state()

    def _create_pca_model(self) -> IncrementalPCA:
        """Create new PCA model with config parameters."""
        return IncrementalPCA(
            n_components=self.config.n_components,
            batch_size=self.config.batch_size,
            whiten=self.config.whiten
        )

    def _load_or_create_state(self) -> dict:
        """Load existing PCA state or create new one."""
        print(f"DEBUG: Checking for PCA state at: {self.pca_state_path}")
        print(f"DEBUG: PCA state file exists: {self.pca_state_path.exists()}")
        if self.pca_state_path.exists():
            print(f"DEBUG: Loading PCA state from: {self.pca_state_path}")
            with open(self.pca_state_path, 'rb') as f:
                state = pickle.load(f)

            # Restore PCA model state
            if state['is_fitted'] and 'pca_object' in state:
                print(f"DEBUG: Restoring PCA object with {state.get('n_samples_seen', 0)} samples from save file")
                trained_images_count = len(state.get('pca_trained_images', set()))
                print(f"DEBUG: PCA was previously trained on {trained_images_count} images")
                self.pca = state['pca_object']
                print(f"DEBUG: After restoration, sklearn n_samples_seen_ = {getattr(self.pca, 'n_samples_seen_', 0)}")

            return state
        else:
            return {
                'n_samples_seen': 0,
                'n_batches_fitted': 0,
                'feature_dim': None,
                'is_fitted': False,
                'pca_trained_images': set()  # Track images that contributed to PCA training
            }

    def _save_state(self) -> None:
        """Save PCA state to single file."""
        if hasattr(self.pca, 'components_'):
            # Just save the sklearn object - deal with version compatibility later
            self.stats['pca_object'] = self.pca

        with open(self.pca_state_path, 'wb') as f:
            pickle.dump(self.stats, f)

    def extract_features_from_detections(self, detections: List[Detection]) -> np.ndarray:
        """Extract valid features from spatial detections for PCA training."""
        all_features = []
        total_patches = 0

        for i, detection in enumerate(detections):
            if detection.features.numel() > 0 and detection.patch_mask.numel() > 0:
                # Features shape: [embed_dim, H_patches, W_patches]
                # Patch mask shape: [H_patches, W_patches]

                # Reshape features to [embed_dim, num_patches]
                embed_dim, h_patches, w_patches = detection.features.shape
                features_flat = detection.features.view(embed_dim, -1)  # [embed_dim, num_patches]

                # Flatten patch mask and ensure it's boolean
                patch_mask_flat = detection.patch_mask.flatten().bool()  # [num_patches]

                # Select only valid patches and transpose to [valid_patches, embed_dim] for PCA
                valid_features = features_flat[:, patch_mask_flat].T

                if valid_features.shape[0] > 0:  # If we have valid patches
                    all_features.append(valid_features.cpu().numpy())
                    total_patches += valid_features.shape[0]
                    if i < 5:  # Only show first 5 detections to avoid spam
                        print(f"  Detection {i}: {valid_features.shape[0]} valid patches")

        if not all_features:
            return np.empty((0, self.stats.get('feature_dim', self.config.n_components)))

        result = np.vstack(all_features)
        print(f"  Total: {total_patches} patches from {len(detections)} detections -> {result.shape[0]} features")
        return result

    def fit_batch(self, detections: List[Detection]) -> bool:
        """Incrementally fit PCA on a batch of detections."""
        # Filter out detections from images already used in PCA training
        new_detections = []
        skipped_images = set()
        new_images_in_batch = set()

        for detection in detections:
            if detection.image_path not in self.stats['pca_trained_images']:
                new_detections.append(detection)
                new_images_in_batch.add(detection.image_path)
            else:
                skipped_images.add(detection.image_path)

        if skipped_images:
            print(f"DEBUG: PCA skipped {len(skipped_images)} images already in training set: {list(skipped_images)[:3]}...")
            print(f"DEBUG: Total PCA trained images: {len(self.stats['pca_trained_images'])}")

        if not new_detections:
            print(f"Skipping PCA fit - all {len(detections)} detections are from already trained images")
            return False

        features = self.extract_features_from_detections(new_detections)

        if features.shape[0] == 0:
            return False

        if self.stats['feature_dim'] is None:
            self.stats['feature_dim'] = features.shape[1]

        print(f"DEBUG: Fitting PCA with {features.shape[0]} features from {len(new_detections)} new detections (skipped {len(detections) - len(new_detections)} from already trained images)")

        print(f"DEBUG: Before partial_fit - sklearn says: {getattr(self.pca, 'n_samples_seen_', 0)} samples seen")
        self.pca.partial_fit(features)
        print(f"DEBUG: After partial_fit - sklearn says: {self.pca.n_samples_seen_} samples seen")

        # Use sklearn's internal counter, not our own
        self.stats['n_samples_seen'] = self.pca.n_samples_seen_
        self.stats['n_batches_fitted'] += 1
        self.stats['is_fitted'] = True

        # Only add images to trained set AFTER successful PCA fitting
        self.stats['pca_trained_images'].update(new_images_in_batch)

        self._save_state()
        return True

    def remove_corrupted_images_from_training(self, corrupted_image_paths: set) -> None:
        """Remove corrupted images from PCA training history."""
        if not corrupted_image_paths:
            return

        removed_count = 0
        for image_path in corrupted_image_paths:
            if image_path in self.stats['pca_trained_images']:
                self.stats['pca_trained_images'].remove(image_path)
                removed_count += 1

        if removed_count > 0:
            print(f"Removed {removed_count} corrupted images from PCA training history")
            self._save_state()

    def transform_features(self, detections: List[Detection]) -> List[np.ndarray]:
        """Transform features using fitted PCA."""
        if not self.stats['is_fitted']:
            raise ValueError("PCA must be fitted before transformation")

        transformed_features = []
        for detection in detections:
            if detection.features.numel() > 0:
                features = detection.features.cpu().numpy().reshape(-1, detection.features.shape[-1])
                transformed = self.pca.transform(features)
                transformed_features.append(transformed)
            else:
                transformed_features.append(np.empty((0, self.config.n_components)))

        return transformed_features

    def should_fit(self, processed_count: int) -> bool:
        """Determine if PCA should be fitted based on processed count."""
        if not self.config.iterative_fitting:
            return False
        return processed_count > 0 and processed_count % self.config.fit_frequency == 0

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio if PCA is fitted."""
        if self.stats['is_fitted'] and hasattr(self.pca, 'explained_variance_ratio_'):
            return self.pca.explained_variance_ratio_
        return None

    def get_stats_summary(self) -> dict:
        """Get summary of PCA statistics."""
        summary = self.stats.copy()
        if self.stats['is_fitted']:
            variance_ratio = self.get_explained_variance_ratio()
            if variance_ratio is not None:
                summary['total_variance_explained'] = float(variance_ratio.sum())
                summary['top_10_components_variance'] = float(variance_ratio[:10].sum())
        return summary

    def is_fitted(self) -> bool:
        """Check if PCA model has been fitted."""
        return self.stats['is_fitted']