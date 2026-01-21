"""RoMa VGG fine feature extraction at grid locations matching DINOv3 patch layout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple

from romatch import roma_outdoor

from .dino_extractor import resize_and_pad_image, get_valid_patch_mask
from ..config.config import RoMaConfig


class RoMaExtractor:
    """Extracts VGG19-BN fine features from RoMa at grid locations matching DINOv3 patch layout.

    RoMa's encoder (CNNandDinov2) contains:
    - self.cnn: VGG19-BN (first 40 layers) - outputs {1, 2, 4, 8} stride features
    - self.dinov2_vitl14: DINOv2 ViT-L/14 - adds stride 16 features (NOT used here)

    VGG19 feature pyramid for 512x512 input:
    - feats[1]: 512x512, 64 channels
    - feats[2]: 256x256, 128 channels
    - feats[4]: 128x128, 256 channels
    - feats[8]: 64x64, 512 channels

    We use stride 8 (64x64) and downsample to 32x32 to match DINOv3 grid.
    """

    def __init__(self, config: RoMaConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.image_size = config.resize_size  # 512 to match DINOv3
        self.patch_size = config.patch_size   # 16 to match DINOv3 grid
        self.n_patches = self.image_size // self.patch_size  # 32

        # Load RoMa model to access VGG19 encoder
        roma_model = roma_outdoor(device=self.device)

        # Extract ONLY the VGG19 part (encoder.cnn), discard DINOv2
        # CNNandDinov2 has: self.cnn (VGG19) and self.dinov2_vitl14 (DINOv2)
        self.vgg = roma_model.encoder.cnn
        self.vgg.eval()

        # Delete everything else to save GPU memory
        del roma_model.encoder.dinov2_vitl14
        del roma_model.decoder
        del roma_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # VGG19 stride 8 features have 512 channels
        self._feature_dim = 512

        # VGG19 only outputs strides {1, 2, 4, 8} - we use 8 and downsample to 32x32
        self._vgg_stride = config.target_stride  # Should be 8 for VGG

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a single image for RoMa (resize to 512, scale to [0,1])."""
        padded_image, _ = resize_and_pad_image(image, self.image_size)
        # RoMa expects [0, 1] range, no ImageNet normalization
        image_tensor = torch.from_numpy(
            np.array(padded_image)
        ).float().permute(2, 0, 1) / 255.0
        return image_tensor

    def _extract_batch_features(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Extract VGG features from a batch of preprocessed images.

        VGG19.forward() returns dict with integer keys {1, 2, 4, 8}:
        - feats[8] = 64x64 for 512x512 input, 512 channels

        Args:
            image_batch: [B, 3, H, W] preprocessed images (scaled to [0, 1])

        Returns:
            features: [B, 512, 32, 32] - downsampled to match DINOv3 grid
        """
        with torch.no_grad():
            # VGG19 returns dict with integer keys {1, 2, 4, 8}
            feature_pyramid = self.vgg(image_batch)

        # Get features at target stride (VGG uses integer keys)
        features = feature_pyramid[self._vgg_stride]

        # Stride 8 gives 64x64 for 512x512 input - downsample to 32x32
        if features.shape[-1] != self.n_patches:
            features = F.adaptive_avg_pool2d(features, (self.n_patches, self.n_patches))

        return features

    @torch.no_grad()
    def extract_features_from_image(
        self,
        image: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features at grid locations matching DINOv3.

        Returns:
            features: [embed_dim, H_patches, W_patches] - 32x32 grid
            valid_mask: [H_patches * W_patches] boolean mask
        """
        padded_image, padding_info = resize_and_pad_image(image, self.image_size)

        image_tensor = torch.from_numpy(
            np.array(padded_image)
        ).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        features = self._extract_batch_features(image_tensor).squeeze(0)

        valid_mask = get_valid_patch_mask(
            padding_info, self.image_size, self.patch_size
        )

        return features.cpu(), valid_mask

    def create_patch_mask_from_image(
        self, image: Image.Image, mask_image: Image.Image
    ) -> torch.Tensor:
        """Create patch-level mask using convolution to detect regions with sufficient content.

        Args:
            image: Input image (unused, kept for interface compatibility)
            mask_image: Grayscale mask image

        Returns:
            valid_patch_mask: [H_patches, W_patches] boolean mask
        """
        mask_tensor = torch.from_numpy(np.array(mask_image)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)

        kernel = torch.ones(1, 1, self.patch_size, self.patch_size)
        mask_patch_sums = F.conv2d(
            mask_tensor.unsqueeze(0),
            kernel,
            stride=self.patch_size
        ).squeeze()

        pixels_per_patch = self.patch_size * self.patch_size
        mask_ratios = mask_patch_sums / pixels_per_patch

        valid_patch_mask = mask_ratios > self.config.black_threshold

        return valid_patch_mask

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

        batch_tensors = []
        batch_masks = []

        for image, mask_image in zip(cropped_images, cropped_masks):
            image_tensor = self._preprocess_image(image)
            batch_tensors.append(image_tensor)

            patch_mask = self.create_patch_mask_from_image(image, mask_image)
            batch_masks.append(patch_mask)

        image_batch = torch.stack(batch_tensors).to(self.device)
        mask_batch = torch.stack(batch_masks).unsqueeze(1)

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
        """Get feature dimension."""
        return self._feature_dim
