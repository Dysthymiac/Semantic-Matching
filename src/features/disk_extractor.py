"""DISK feature extractor with confidence-weighted descriptors.

Uses kornia's DISK implementation to extract dense learned features with confidence
scores. Features are max-pooled by score within semantic grid cells, then scaled
by confidence for attention-like weighting in downstream Fisher Vector encoding.

IMPORTANT for spatial alignment:
- Uses same resize_and_pad_image as DINO (identical spatial layout)
- DISK expects [0, 1] range (NOT ImageNet normalized like DINO)
- Same 32x32 patch grid for 512px input with 16px patches
"""

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config.config import DISKConfig
# Reuse DINO's resize/pad for exact spatial alignment
from .dino_extractor import resize_and_pad_image, get_valid_patch_mask


class DISKExtractor:
    """DISK feature extractor with confidence-weighted descriptors aligned to semantic grid."""

    def __init__(self, config: Optional[DISKConfig] = None, device: str = "cuda"):
        self.config = config or DISKConfig()
        self.device = torch.device(device)
        self.resize_size = self.config.resize_size
        self.patch_size = self.config.patch_size
        self.n_patches = self.resize_size // self.patch_size  # 32

        try:
            from kornia.feature import DISK
        except ImportError:
            raise ImportError("kornia required. Install: pip install kornia")

        self.model = DISK.from_pretrained(self.config.pretrained_weights).to(self.device)
        self.model.eval()
        self._feature_dim = 128

    def get_feature_dim(self) -> int:
        return self._feature_dim

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor in [0, 1] range (DISK's expected input)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Convert to [0, 1] float32 tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        # [H, W, 3] -> [1, 3, H, W]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def _maxpool_by_score(
        self, scores: torch.Tensor, descriptors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Max-pool descriptors by score within semantic grid cells.

        For each patch_size x patch_size region, select the descriptor
        from the location with highest score.
        """
        B, _, H, W = scores.shape
        D = descriptors.shape[1]
        ps = self.patch_size

        # Unfold into patches
        scores_patches = scores.unfold(2, ps, ps).unfold(3, ps, ps)
        desc_patches = descriptors.unfold(2, ps, ps).unfold(3, ps, ps)

        # Reshape: [B, n_patches, n_patches, ps*ps]
        scores_flat = scores_patches.reshape(B, self.n_patches, self.n_patches, ps * ps)

        # Find max score position
        max_idx = scores_flat.argmax(dim=-1)
        max_scores = scores_flat.gather(-1, max_idx.unsqueeze(-1)).squeeze(-1)

        # Reshape descriptors: [B, D, n_patches, n_patches, ps*ps]
        desc_flat = desc_patches.reshape(B, D, self.n_patches, self.n_patches, ps * ps)

        # Select descriptor at max score position
        max_idx_expanded = max_idx.unsqueeze(1).unsqueeze(-1).expand(B, D, -1, -1, 1)
        selected_desc = desc_flat.gather(-1, max_idx_expanded).squeeze(-1)

        return selected_desc, max_scores.unsqueeze(1)

    @torch.no_grad()
    def extract_features_from_image(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract DISK features from a single image.

        Uses same resize/pad as DINO for spatial alignment.

        Pipeline:
        1. Resize and pad (same as DINO)
        2. Convert to [0, 1] tensor (DISK's expected input)
        3. Extract dense DISK descriptors and scores
        4. L2 normalize descriptors to unit norm
        5. Max-pool by score to 32x32 grid
        6. Scale unit-norm descriptors by score (magnitude = confidence)

        Returns:
            - features: [128, 32, 32] confidence-weighted descriptors
            - valid_mask: [1024] boolean
        """
        # Use DINO's preprocessing for exact spatial alignment
        padded_image, padding_info = resize_and_pad_image(image, self.resize_size)

        # Convert to [0, 1] tensor (DISK's expected input, NOT ImageNet normalized)
        img_tensor = self._image_to_tensor(padded_image)

        # Extract dense descriptors and scores
        scores, descriptors = self.model.heatmap_and_dense_descriptors(img_tensor)

        # L2 normalize descriptors to unit norm
        descriptors = F.normalize(descriptors, p=2, dim=1)

        # Max-pool by score to align with semantic grid
        selected_desc, selected_scores = self._maxpool_by_score(scores, descriptors)

        # Re-normalize selected descriptors to unit norm
        selected_desc = F.normalize(selected_desc, p=2, dim=1)

        # Scale by score: magnitude now equals confidence
        weighted_desc = selected_desc * selected_scores

        features = weighted_desc.squeeze(0).cpu()  # [128, 32, 32]

        # Use DINO's mask function for consistency
        valid_mask = get_valid_patch_mask(padding_info, self.resize_size, self.patch_size)

        return features, valid_mask

    def create_patch_mask_from_image(
        self, image: Image.Image, mask_image: Image.Image
    ) -> torch.Tensor:
        """Create patch-level mask using same logic as DINO."""
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")

        mask_tensor = torch.from_numpy(np.array(mask_image)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]

        kernel = torch.ones(1, 1, self.patch_size, self.patch_size)
        mask_patch_sums = F.conv2d(
            mask_tensor.unsqueeze(0),
            kernel,
            stride=self.patch_size
        ).squeeze()

        pixels_per_patch = self.patch_size * self.patch_size
        mask_ratios = mask_patch_sums / pixels_per_patch

        return mask_ratios > self.config.black_threshold

    @torch.no_grad()
    def extract_features_with_masks(
        self,
        cropped_images: List[Image.Image],
        cropped_masks: List[Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract DISK features from batch of cropped images with masks.

        Args:
            cropped_images: List of PIL Images (already 512x512)
            cropped_masks: List of PIL mask images

        Returns:
            - features_batch: [B, 128, 32, 32]
            - mask_batch: [B, 1, 32, 32]
        """
        features_list = []
        masks_list = []

        for image, mask in zip(cropped_images, cropped_masks):
            # Convert to [0, 1] tensor
            img_tensor = self._image_to_tensor(image)

            # Extract dense features
            scores, descriptors = self.model.heatmap_and_dense_descriptors(img_tensor)

            # L2 normalize
            descriptors = F.normalize(descriptors, p=2, dim=1)

            # Max-pool by score
            selected_desc, selected_scores = self._maxpool_by_score(scores, descriptors)

            # Re-normalize and scale
            selected_desc = F.normalize(selected_desc, p=2, dim=1)
            weighted_desc = selected_desc * selected_scores

            features_list.append(weighted_desc.squeeze(0).cpu())

            # Create patch mask
            patch_mask = self.create_patch_mask_from_image(image, mask)
            masks_list.append(patch_mask.unsqueeze(0).float())

        features_batch = torch.stack(features_list, dim=0)
        mask_batch = torch.stack(masks_list, dim=0)

        return features_batch, mask_batch
