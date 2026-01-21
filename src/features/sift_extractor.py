"""Dense SIFT feature extractor for texture encoding.

Extracts SIFT descriptors at fixed grid positions matching the 32×32 patch grid
used by DINOv3/DeDoDe/RoMa for spatial alignment.

Standard Dense SIFT + Fisher Vector pipeline uses multi-scale extraction:
- Extract 128D SIFT at each scale (e.g., 16, 24, 32 pixels)
- Concatenate to get 128 * n_scales dimensions per keypoint
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config.config import SIFTConfig


def make_grid_keypoints(
    image_size: int = 512,
    patch_size: int = 16,
    scale: float = 12.0,
) -> List[cv2.KeyPoint]:
    """
    Generate OpenCV KeyPoints at patch centers matching DINOv3 spatial layout.

    For 512×512 image with patch_size=16:
    - 32×32 = 1024 keypoints at patch centers
    - Centers at: 8, 24, 40, 56, ... 504 pixels

    Args:
        image_size: Input image size (assumes square)
        patch_size: Patch size in pixels
        scale: Keypoint scale (affects descriptor region size)

    Returns:
        List of cv2.KeyPoint objects
    """
    n_patches = image_size // patch_size  # 32

    # Patch centers in pixel coordinates
    centers = np.arange(patch_size // 2, image_size, patch_size, dtype=np.float32)
    # centers = [8, 24, 40, ..., 504]

    keypoints = []
    for y in centers:
        for x in centers:
            # KeyPoint(x, y, size, angle, response, octave, class_id)
            kp = cv2.KeyPoint(x=float(x), y=float(y), size=scale)
            keypoints.append(kp)

    return keypoints  # 1024 keypoints for 32×32 grid


class SIFTExtractor:
    """Dense multi-scale SIFT feature extractor aligned with DINOv3 patch grid."""

    def __init__(self, config: Optional[SIFTConfig] = None, device: str = "cpu"):
        """
        Initialize Dense SIFT extractor with multi-scale support.

        Args:
            config: SIFT configuration
            device: Ignored (SIFT runs on CPU via OpenCV)
        """
        self.config = config or SIFTConfig()
        self.resize_size = self.config.resize_size
        self.patch_size = self.config.patch_size
        self.n_patches = self.resize_size // self.patch_size  # 32

        # Get scales from config
        self.scales = self.config.sift_scales
        self.n_scales = len(self.scales)

        # Create SIFT detector with settings for dense extraction
        self.sift = cv2.SIFT_create(
            contrastThreshold=self.config.contrast_threshold,
            edgeThreshold=self.config.edge_threshold,
        )

        # Pre-compute grid keypoints for each scale
        self.grid_keypoints_per_scale = {
            scale: make_grid_keypoints(
                image_size=self.resize_size,
                patch_size=self.patch_size,
                scale=scale,
            )
            for scale in self.scales
        }

        # Feature dimension: 128D per scale, concatenated
        self._feature_dim = 128 * self.n_scales

    def get_feature_dim(self) -> int:
        """Return feature dimension (128 * n_scales for multi-scale SIFT)."""
        return self._feature_dim

    def _preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Preprocess image: resize with aspect preservation and pad to square.

        Args:
            image: PIL Image

        Returns:
            - Grayscale numpy array [H, W] uint8
            - Padding info (left, top, right, bottom)
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        orig_w, orig_h = image.size

        # Compute scale to fit in resize_size while preserving aspect ratio
        scale = self.resize_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        image_resized = image.resize((new_w, new_h), Image.BILINEAR)

        # Compute padding to center the image
        pad_left = (self.resize_size - new_w) // 2
        pad_top = (self.resize_size - new_h) // 2
        pad_right = self.resize_size - new_w - pad_left
        pad_bottom = self.resize_size - new_h - pad_top

        # Create padded image (black background)
        padded = Image.new('RGB', (self.resize_size, self.resize_size), (0, 0, 0))
        padded.paste(image_resized, (pad_left, pad_top))

        # Convert to grayscale numpy array for SIFT
        gray = np.array(padded.convert('L'))

        return gray, (pad_left, pad_top, pad_right, pad_bottom)

    def _extract_sift_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale SIFT descriptors at grid keypoints.

        Args:
            gray_image: Grayscale image [H, W] uint8

        Returns:
            Descriptors [N_keypoints, 128 * n_scales] float32
        """
        n_keypoints = self.n_patches * self.n_patches  # 1024
        all_descriptors = []

        for scale in self.scales:
            keypoints = self.grid_keypoints_per_scale[scale]
            _, descriptors = self.sift.compute(gray_image, keypoints)

            if descriptors is None:
                # Return zeros if SIFT fails (shouldn't happen with our settings)
                descriptors = np.zeros((n_keypoints, 128), dtype=np.float32)
            else:
                # RootSIFT (Arandjelović & Zisserman, 2012):
                # 1. L1 normalize
                # 2. Element-wise square root
                # 3. L2 normalize
                # This significantly improves matching performance
                descriptors = descriptors.astype(np.float32)

                # L1 normalize
                l1_norms = np.sum(np.abs(descriptors), axis=1, keepdims=True)
                l1_norms = np.maximum(l1_norms, 1e-7)
                descriptors = descriptors / l1_norms

                # Element-wise square root (RootSIFT)
                descriptors = np.sqrt(descriptors)

                # L2 normalize
                l2_norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
                l2_norms = np.maximum(l2_norms, 1e-7)
                descriptors = descriptors / l2_norms

            all_descriptors.append(descriptors)

        # Concatenate descriptors from all scales: [N_keypoints, 128 * n_scales]
        return np.concatenate(all_descriptors, axis=1)

    def _get_valid_patch_mask(
        self,
        padding_info: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        Create mask for patches that don't overlap with padding areas.

        Args:
            padding_info: (left, top, right, bottom) padding in pixels

        Returns:
            Boolean mask [H_patches, W_patches]
        """
        valid_mask = torch.ones(
            (self.n_patches, self.n_patches), dtype=torch.bool
        )

        left_pad, top_pad, right_pad, bottom_pad = padding_info

        # Mark patches that overlap with padding as invalid
        left_invalid = left_pad // self.patch_size
        top_invalid = top_pad // self.patch_size
        right_invalid = right_pad // self.patch_size if right_pad > 0 else 0
        bottom_invalid = bottom_pad // self.patch_size if bottom_pad > 0 else 0

        if left_invalid > 0:
            valid_mask[:, :left_invalid] = False
        if top_invalid > 0:
            valid_mask[:top_invalid, :] = False
        if right_invalid > 0:
            valid_mask[:, -right_invalid:] = False
        if bottom_invalid > 0:
            valid_mask[-bottom_invalid:, :] = False

        return valid_mask

    @torch.no_grad()
    def extract_features_from_image(
        self,
        image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract Dense multi-scale SIFT features from a single image.

        Args:
            image: PIL Image

        Returns:
            - features: [embed_dim, H_patches, W_patches] = [128*n_scales, 32, 32]
            - valid_mask: [H_patches * W_patches] = [1024] boolean
        """
        # Preprocess
        gray, padding_info = self._preprocess_image(image)

        # Extract multi-scale SIFT
        descriptors = self._extract_sift_features(gray)  # [1024, 128*n_scales]

        # Reshape to spatial format [D, H, W]
        features = descriptors.reshape(
            self.n_patches, self.n_patches, self._feature_dim
        )
        features = np.transpose(features, (2, 0, 1))  # [128*n_scales, 32, 32]
        features = torch.from_numpy(features).float()

        # Get padding mask
        valid_mask = self._get_valid_patch_mask(padding_info)

        return features, valid_mask.flatten()

    def create_patch_mask_from_image(
        self,
        image: Image.Image,
        mask_image: Image.Image
    ) -> torch.Tensor:
        """
        Create patch-level mask combining padding and content masks.

        Args:
            image: Original PIL Image (for aspect ratio)
            mask_image: Binary mask PIL Image (white = object)

        Returns:
            Boolean mask [H_patches, W_patches] = [32, 32]
        """
        # Get padding mask from image aspect ratio
        _, padding_info = self._preprocess_image(image)
        padding_mask = self._get_valid_patch_mask(padding_info)

        # Process content mask
        if mask_image.mode != 'L':
            mask_image = mask_image.convert('L')

        # Resize mask to match preprocessing
        orig_w, orig_h = image.size
        scale = self.resize_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        mask_resized = mask_image.resize((new_w, new_h), Image.NEAREST)

        # Pad mask
        pad_left = (self.resize_size - new_w) // 2
        pad_top = (self.resize_size - new_h) // 2

        padded_mask = Image.new('L', (self.resize_size, self.resize_size), 0)
        padded_mask.paste(mask_resized, (pad_left, pad_top))

        # Convert to tensor and compute patch-wise content
        mask_tensor = torch.from_numpy(np.array(padded_mask)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Use convolution to compute mean mask value per patch
        kernel = torch.ones(1, 1, self.patch_size, self.patch_size)
        kernel = kernel / (self.patch_size * self.patch_size)

        patch_means = F.conv2d(mask_tensor, kernel, stride=self.patch_size)
        patch_means = patch_means.squeeze()  # [32, 32]

        # Patch is valid if content ratio > threshold
        content_mask = patch_means > self.config.black_threshold

        # Combine padding and content masks
        combined_mask = padding_mask & content_mask

        return combined_mask

    @torch.no_grad()
    def extract_features_with_masks(
        self,
        cropped_images: List[Image.Image],
        cropped_masks: List[Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract multi-scale SIFT features from a batch of cropped images with masks.

        Args:
            cropped_images: List of PIL Images (already cropped and padded to 512×512)
            cropped_masks: List of PIL mask images

        Returns:
            - features_batch: [B, embed_dim, H_patches, W_patches] = [B, 128*n_scales, 32, 32]
            - mask_batch: [B, 1, H_patches, W_patches] = [B, 1, 32, 32]
        """
        features_list = []
        masks_list = []

        for image, mask in zip(cropped_images, cropped_masks):
            # Convert image to grayscale numpy
            if image.mode != 'L':
                gray = np.array(image.convert('L'))
            else:
                gray = np.array(image)

            # Extract multi-scale SIFT features
            descriptors = self._extract_sift_features(gray)  # [1024, 128*n_scales]

            # Reshape to spatial format
            features = descriptors.reshape(
                self.n_patches, self.n_patches, self._feature_dim
            )
            features = np.transpose(features, (2, 0, 1))  # [128*n_scales, 32, 32]
            features_list.append(torch.from_numpy(features).float())

            # Compute patch mask from content
            patch_mask = self._compute_content_mask(mask)  # [32, 32]
            masks_list.append(patch_mask.unsqueeze(0).float())  # [1, 32, 32]

        # Stack into batches
        features_batch = torch.stack(features_list, dim=0)  # [B, 128*n_scales, 32, 32]
        mask_batch = torch.stack(masks_list, dim=0)  # [B, 1, 32, 32]

        return features_batch, mask_batch

    def _compute_content_mask(self, mask_image: Image.Image) -> torch.Tensor:
        """
        Compute patch-level content mask from a mask image.

        Args:
            mask_image: PIL mask image (already resized to 512×512)

        Returns:
            Boolean mask [32, 32]
        """
        if mask_image.mode != 'L':
            mask_image = mask_image.convert('L')

        mask_array = np.array(mask_image)
        mask_tensor = torch.from_numpy(mask_array).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Compute mean per patch
        kernel = torch.ones(1, 1, self.patch_size, self.patch_size)
        kernel = kernel / (self.patch_size * self.patch_size)

        patch_means = F.conv2d(mask_tensor, kernel, stride=self.patch_size)
        patch_means = patch_means.squeeze()  # [32, 32]

        # Threshold
        content_mask = patch_means > self.config.black_threshold

        return content_mask
