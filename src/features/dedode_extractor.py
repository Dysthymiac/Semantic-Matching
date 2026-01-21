"""DeDoDe dense descriptor extraction at grid locations matching DINOv3 patch layout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple

from kornia.feature import DeDoDe

from .dino_extractor import resize_and_pad_image, get_valid_patch_mask
from ..config.config import DeDoDeConfig


def make_patch_grid_keypoints(
    image_size: int = 512,
    patch_size: int = 16,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Generate keypoints at patch centers matching DINOv3 spatial layout.

    Returns keypoints in [-1, 1] normalized coordinates for Kornia DeDoDe.
    Shape: [1, N, 2] where N = (image_size/patch_size)^2
    """
    n_patches = image_size // patch_size

    # Patch centers in pixel coordinates
    centers = torch.arange(patch_size // 2, image_size, patch_size, dtype=torch.float32)

    # Create grid of (x, y) coordinates
    yy, xx = torch.meshgrid(centers, centers, indexing='ij')

    # Normalize to [-1, 1] (Kornia convention)
    xx_norm = (xx / (image_size - 1)) * 2 - 1
    yy_norm = (yy / (image_size - 1)) * 2 - 1

    # Stack as [N, 2] then add batch dim
    keypoints = torch.stack([xx_norm.flatten(), yy_norm.flatten()], dim=-1)

    return keypoints.unsqueeze(0).to(device)  # [1, N, 2]


class DeDoDeExtractor:
    """Extracts DeDoDe descriptors at grid locations matching DINOv3 patch layout."""

    def __init__(self, config: DeDoDeConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.image_size = config.resize_size  # Target crop size (512 to match DINOv3)
        self.patch_size = config.patch_size   # 16 to match DINOv3 grid
        self.n_patches = self.image_size // self.patch_size  # 32x32 grid

        # DINOv2 (used by G model) requires input multiple of 14
        # Pad images to nearest multiple of 14 >= image_size
        self._dinov2_size = ((self.image_size + 13) // 14) * 14  # 512 -> 518

        # Load DeDoDe - must load with detector weights even though we only use descriptor
        self.model = DeDoDe.from_pretrained(
            detector_weights="L-upright",  # Required by API, but unused
            descriptor_weights=config.descriptor_weights,
        ).to(self.device)
        self.model.eval()

        # Delete detector to save GPU memory (~300MB for L-upright)
        # describe() only uses self.descriptor and self.normalizer
        del self.model.detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Pre-compute grid keypoints at DINOv3 patch centers
        # Keypoints are in [-1, 1] normalized coords relative to padded image
        self.grid_keypoints = self._make_grid_keypoints_for_padded_image()

        # Feature dimension (G model = 512, B model = 256)
        self._feature_dim = 512 if 'G' in config.descriptor_weights else 256

    def _make_grid_keypoints_for_padded_image(self) -> torch.Tensor:
        """Generate keypoints at DINOv3 patch centers, normalized to padded image coords."""
        # Patch centers in pixel coords (relative to original image_size)
        centers = torch.arange(
            self.patch_size // 2, self.image_size, self.patch_size, dtype=torch.float32
        )

        # Create grid of (x, y) coordinates
        yy, xx = torch.meshgrid(centers, centers, indexing='ij')

        # Normalize to [-1, 1] relative to PADDED image size (for Kornia)
        # The padding is added to the right/bottom, so original content is at top-left
        xx_norm = (xx / (self._dinov2_size - 1)) * 2 - 1
        yy_norm = (yy / (self._dinov2_size - 1)) * 2 - 1

        keypoints = torch.stack([xx_norm.flatten(), yy_norm.flatten()], dim=-1)
        return keypoints.unsqueeze(0).to(self.device)  # [1, N, 2]

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a single image for DeDoDe (resize to 512, pad to 518).

        Note: ImageNet normalization is handled by DeDoDe's describe() method.
        """
        # First resize/pad to match DINOv3 size (512)
        padded_image, _ = resize_and_pad_image(image, self.image_size)
        image_tensor = torch.from_numpy(
            np.array(padded_image)
        ).float().permute(2, 0, 1) / 255.0

        # Pad to DINOv2-compatible size (518) with zeros on right/bottom
        pad_amount = self._dinov2_size - self.image_size  # 6 pixels
        if pad_amount > 0:
            image_tensor = F.pad(image_tensor, (0, pad_amount, 0, pad_amount), value=0)

        return image_tensor

    def _extract_batch_features(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of preprocessed images.

        Args:
            image_batch: [B, 3, H, W] preprocessed images

        Returns:
            features: [B, embed_dim, H_patches, W_patches]
        """
        batch_size = image_batch.shape[0]

        # Expand keypoints for batch: [1, N, 2] -> [B, N, 2]
        batch_keypoints = self.grid_keypoints.expand(batch_size, -1, -1)

        # Extract descriptors - returns tensor [B, N, D] directly
        features = self.model.describe(image_batch, batch_keypoints)

        # Reshape: [B, N, D] -> [B, D, H, W]
        features = features.view(
            batch_size, self.n_patches, self.n_patches, -1
        ).permute(0, 3, 1, 2)  # [B, D, H, W]

        return features

    @torch.no_grad()
    def extract_features_from_image(
        self,
        image: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract DeDoDe descriptors at grid locations matching DINOv3.

        Returns:
            features: [embed_dim, H_patches, W_patches] - same 32x32 grid as DINOv3
            valid_mask: [H_patches * W_patches] boolean mask
        """
        # Resize/pad to 512 (same as DINOv3)
        padded_image, padding_info = resize_and_pad_image(image, self.image_size)

        # Convert to tensor (normalization handled by DeDoDe's describe())
        image_tensor = torch.from_numpy(
            np.array(padded_image)
        ).float().permute(2, 0, 1) / 255.0

        # Pad to DINOv2-compatible size (518)
        pad_amount = self._dinov2_size - self.image_size
        if pad_amount > 0:
            image_tensor = F.pad(image_tensor, (0, pad_amount, 0, pad_amount), value=0)

        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # Extract features
        features = self._extract_batch_features(image_tensor).squeeze(0)  # [D, H, W]

        # Get valid patch mask (reuse DINOv3 logic)
        valid_mask = get_valid_patch_mask(
            padding_info, self.image_size, self.patch_size
        )

        return features.cpu(), valid_mask

    def create_patch_mask_from_image(self, image: Image.Image, mask_image: Image.Image) -> torch.Tensor:
        """Create patch-level mask using convolution to detect regions with sufficient content.

        Args:
            image: Input image (unused, kept for interface compatibility)
            mask_image: Grayscale mask image

        Returns:
            valid_patch_mask: [H_patches, W_patches] boolean mask
        """
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(np.array(mask_image)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]

        # Use convolution to count non-black pixels per patch
        kernel = torch.ones(1, 1, self.patch_size, self.patch_size)
        mask_patch_sums = F.conv2d(
            mask_tensor.unsqueeze(0),  # [1, 1, H, W]
            kernel,
            stride=self.patch_size
        ).squeeze()  # [H_patches, W_patches]

        # Calculate ratio of masked pixels per patch
        pixels_per_patch = self.patch_size * self.patch_size
        mask_ratios = mask_patch_sums / pixels_per_patch

        # Patch is valid if mask ratio > threshold
        valid_patch_mask = mask_ratios > self.config.black_threshold

        return valid_patch_mask  # [H_patches, W_patches]

    @torch.no_grad()
    def extract_features_with_masks(
        self,
        cropped_images: List[Image.Image],
        cropped_masks: List[Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from cropped images and create unified patch masks.

        Args:
            cropped_images: List of cropped detection images
            cropped_masks: List of corresponding mask images

        Returns:
            features_batch: [B, embed_dim, H_patches, W_patches]
            mask_batch: [B, 1, H_patches, W_patches]
        """
        if not cropped_images:
            return (
                torch.empty((0, self._feature_dim, 0, 0)),
                torch.empty((0, 1, 0, 0))
            )

        # Process images and masks
        batch_tensors = []
        batch_masks = []

        for image, mask_image in zip(cropped_images, cropped_masks):
            image_tensor = self._preprocess_image(image)
            batch_tensors.append(image_tensor)

            patch_mask = self.create_patch_mask_from_image(image, mask_image)
            batch_masks.append(patch_mask)

        # Stack into batches
        image_batch = torch.stack(batch_tensors).to(self.device)  # [B, 3, H, W]
        mask_batch = torch.stack(batch_masks).unsqueeze(1)  # [B, 1, H_patches, W_patches]

        # Process in sub-batches if needed
        batch_size = self.config.batch_size
        all_features = []

        for i in range(0, len(image_batch), batch_size):
            sub_batch = image_batch[i:i + batch_size]
            features = self._extract_batch_features(sub_batch)
            all_features.append(features.cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        features_spatial = torch.cat(all_features, dim=0)

        return features_spatial, mask_batch.float()

    def get_feature_dim(self) -> int:
        """Get descriptor dimension."""
        return self._feature_dim
