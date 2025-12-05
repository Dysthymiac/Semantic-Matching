"""DINOv3 feature extraction functionality."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms import v2, InterpolationMode
from typing import List, Tuple

from ..config.config import DINOConfig
from .dinov3_loader import load_dinov3_model


def make_dino_transform_with_padding(target_size: int = 224) -> v2.Compose:
    """Create DINOv3 preprocessing transform preserving aspect ratio with padding."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


def resize_and_pad_image(image: Image.Image, target_size: int = 512) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Resize image preserving aspect ratio and pad to square, returning padding info."""
    original_width, original_height = image.size

    # Calculate scaling factor to fit within target_size while preserving aspect ratio
    scale = min(target_size / original_width, target_size / original_height)

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize with high quality interpolation
    resized_image = image.resize((new_width, new_height), Image.BICUBIC)

    # Calculate padding to center the image
    pad_width = (target_size - new_width) // 2
    pad_height = (target_size - new_height) // 2

    # Add padding to make it square (pad with black)
    padded_image = ImageOps.expand(
        resized_image,
        (pad_width, pad_height, target_size - new_width - pad_width, target_size - new_height - pad_height),
        fill=(0, 0, 0)
    )

    # Return image and padding info (left, top, right, bottom)
    padding_info = (
        pad_width,
        pad_height,
        target_size - new_width - pad_width,
        target_size - new_height - pad_height
    )

    return padded_image, padding_info


def get_valid_patch_mask(padding_info: Tuple[int, int, int, int], image_size: int = 224, patch_size: int = 16) -> torch.Tensor:
    """Create mask for patches that don't overlap with padding areas."""
    left_pad, top_pad, right_pad, bottom_pad = padding_info

    # Calculate number of patches
    num_patches_per_side = image_size // patch_size

    # Create 2D mask for valid patches
    valid_mask = torch.ones((num_patches_per_side, num_patches_per_side), dtype=torch.bool)

    # Calculate which patches are affected by padding
    left_invalid_patches = left_pad // patch_size
    top_invalid_patches = top_pad // patch_size
    right_invalid_patches = right_pad // patch_size if right_pad > 0 else 0
    bottom_invalid_patches = bottom_pad // patch_size if bottom_pad > 0 else 0

    # Mark invalid patches (those overlapping with padding)
    if left_invalid_patches > 0:
        valid_mask[:, :left_invalid_patches] = False
    if top_invalid_patches > 0:
        valid_mask[:top_invalid_patches, :] = False
    if right_invalid_patches > 0:
        valid_mask[:, -right_invalid_patches:] = False
    if bottom_invalid_patches > 0:
        valid_mask[-bottom_invalid_patches:, :] = False

    return valid_mask.flatten()


class DINOv3Extractor:
    """Handles DINOv3 feature extraction from image patches with aspect ratio preservation."""

    def __init__(self, config: DINOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = load_dinov3_model(config.model_name, config.device, config.weights_file_path)
        self.transform = make_dino_transform_with_padding(config.resize_size)
        self.patch_size = 16  # Standard DINOv3 patch size

        # Model layer configuration
        self.model_to_num_layers = {
            'dinov3_vits16': 12,
            'dinov3_vitb16': 12,
            'dinov3_vitl16': 24,
            'dinov3_vith16plus': 32,
            'dinov3_vit7b16': 40,
        }
        self.n_layers = self.model_to_num_layers.get(config.model_name, 12)
        self.last_layer = self.n_layers - 1
        self.extraction_layer = config.extraction_layer if config.extraction_layer is not None else self.last_layer

    def _extract_patch_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract patch features using DINOv3 API, returning [batch, embed_dim, h_patches, w_patches]."""
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            feats = self.model.get_intermediate_layers(
                image_tensor,
                n=[self.extraction_layer],
                reshape=True,
                norm=True
            )
            # Extract features: [B, embed_dim, H/patch_size, W/patch_size]
            features = feats[0].detach().cpu()
            batch_size, embed_dim, h_patches, w_patches = features.shape
            num_patches = h_patches * w_patches
            print(f"DINOv3 batch: {batch_size} images, {h_patches}x{w_patches}={num_patches} patches, {embed_dim}D features")

            # Keep spatial format: [batch, embed_dim, h_patches, w_patches]
            return features

    @torch.no_grad()
    def extract_features_from_image(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract DINOv3 features from image, returning features and valid patch mask."""
        # Resize and pad image
        padded_image, padding_info = resize_and_pad_image(image, self.config.resize_size)

        # Apply transform
        image_tensor = self.transform(padded_image).unsqueeze(0).to(self.device)

        # Extract patch features using reusable method
        patch_features_batch = self._extract_patch_features(image_tensor)
        patch_features = patch_features_batch.squeeze(0)  # Remove batch dimension

        # Get valid patch mask
        valid_mask = get_valid_patch_mask(padding_info, self.config.resize_size, self.patch_size)

        return patch_features, valid_mask

    @torch.no_grad()
    def extract_features_batch(self, images: List[Image.Image]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract DINOv3 features from batch of images."""
        if not images:
            return torch.empty((0, self.get_feature_dim())), []

        batch_tensors = []
        valid_masks = []

        # Process each image with padding
        for image in images:
            padded_image, padding_info = resize_and_pad_image(image, self.config.resize_size)
            image_tensor = self.transform(padded_image)
            batch_tensors.append(image_tensor)
            valid_mask = get_valid_patch_mask(padding_info, self.config.resize_size, self.patch_size)
            valid_masks.append(valid_mask)

        # Stack into batch
        image_tensors = torch.stack(batch_tensors).to(self.device)

        # Process in smaller batches if needed
        batch_size = self.config.batch_size
        all_features = []

        for i in range(0, len(image_tensors), batch_size):
            batch = image_tensors[i:i + batch_size]
            features = self._extract_patch_features(batch)
            all_features.append(features)

        return torch.cat(all_features, dim=0), valid_masks

    def extract_features_from_masked_regions_batch(self, images: List[Image.Image], masks_list: List[torch.Tensor]) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Extract features from masked regions across a batch of images."""
        all_masked_images = []
        image_indices = []  # Track which original image each masked image belongs to
        mask_indices = []   # Track which mask within that image

        # Collect all masked images from all input images
        for img_idx, (image, masks) in enumerate(zip(images, masks_list)):
            for mask_idx, mask in enumerate(masks):
                if mask.sum() == 0:
                    continue

                mask_pil = Image.fromarray((mask.cpu().numpy() * 255).astype('uint8'), mode='L')
                black_background = Image.new('RGB', image.size, (0, 0, 0))
                masked_image = Image.composite(image, black_background, mask_pil)
                all_masked_images.append(masked_image)
                image_indices.append(img_idx)
                mask_indices.append(mask_idx)

        if not all_masked_images:
            return [(torch.empty((0, self.get_feature_dim())), []) for _ in images]

        # Process all masked images in one batch
        all_features, all_valid_masks = self.extract_features_batch(all_masked_images)

        # Group results back by original image
        results = [[] for _ in images]
        for i, (img_idx, mask_idx) in enumerate(zip(image_indices, mask_indices)):
            features = all_features[i]
            valid_mask = all_valid_masks[i]
            if img_idx >= len(results):
                results.extend([[] for _ in range(img_idx - len(results) + 1)])
            results[img_idx].append((features, valid_mask))

        # Convert to expected format
        final_results = []
        for img_idx, image_results in enumerate(results):
            if not image_results:
                final_results.append((torch.empty((0, self.get_feature_dim())), []))
            else:
                # Concatenate features from all masks for this image
                img_features = torch.cat([feat for feat, _ in image_results], dim=0)
                img_valid_masks = [mask for _, mask in image_results]
                final_results.append((img_features, img_valid_masks))

        return final_results

    def extract_features_from_masked_regions(self, image: Image.Image, masks: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract features from masked regions of an image - wrapper for batch method."""
        results = self.extract_features_from_masked_regions_batch([image], [masks])
        return results[0]

    def create_patch_mask_from_image(self, image: Image.Image, mask_image: Image.Image) -> torch.Tensor:
        """Create patch-level mask using convolution to efficiently detect regions with sufficient content."""
        # Convert images to tensors
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0  # [H, W, 3]
        mask_tensor = torch.from_numpy(np.array(mask_image)).float() / 255.0  # [H, W]

        # Rearrange to [C, H, W] format
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]

        # Calculate patch size in pixels
        h, w = image_tensor.shape[0], image_tensor.shape[1]
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size

        # Use convolution to count non-black pixels per patch (similar to DINOv3 notebooks)
        # Create kernel to sum pixels in each patch
        kernel = torch.ones(1, 1, self.patch_size, self.patch_size)

        # Apply convolution with stride=patch_size to get patch-wise sums
        mask_patch_sums = F.conv2d(
            mask_tensor.unsqueeze(0),  # [1, 1, H, W]
            kernel,
            stride=self.patch_size
        ).squeeze()  # [H_patches, W_patches]

        # Calculate total pixels per patch
        pixels_per_patch = self.patch_size * self.patch_size

        # Calculate ratio of masked (non-black) pixels per patch
        mask_ratios = mask_patch_sums / pixels_per_patch

        # Create valid patch mask: patch is valid if mask ratio > threshold
        # (meaning it contains enough actual object content, not black background)
        valid_patch_mask = mask_ratios > self.config.black_threshold

        return valid_patch_mask  # [H_patches, W_patches]

    def extract_features_with_masks(self, cropped_images: List[Image.Image], cropped_masks: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from cropped images and create unified patch masks."""
        if not cropped_images:
            return torch.empty((0, self.get_feature_dim(), 0, 0)), torch.empty((0, 1, 0, 0))

        # Convert to tensor batch
        batch_tensors = []
        batch_masks = []

        for image, mask_image in zip(cropped_images, cropped_masks):
            # Process image for DINOv3
            image_tensor = self.transform(image)
            batch_tensors.append(image_tensor)

            # Create patch-level mask
            patch_mask = self.create_patch_mask_from_image(image, mask_image)
            batch_masks.append(patch_mask)

        # Stack into batches
        image_batch = torch.stack(batch_tensors).to(self.device)  # [B, 3, H, W]
        mask_batch = torch.stack(batch_masks).unsqueeze(1)  # [B, 1, H_patches, W_patches]

        # Extract features in spatial format
        features_batch = self._extract_patch_features(image_batch)  # [B, embed_dim, H_patches, W_patches]

        # Move back to CPU and return both in spatial format
        return features_batch.cpu(), mask_batch.float()

    def get_feature_dim(self) -> int:
        """Get the feature dimension for the current model."""
        model_dims = {
            'dinov3_vits16': 384,
            'dinov3_vitb16': 768,
            'dinov3_vitl16': 1024,
            'dinov3_vith16': 1280
        }
        return model_dims.get(self.config.model_name, 768)