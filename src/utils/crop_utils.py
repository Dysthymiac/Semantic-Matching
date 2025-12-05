"""Utilities for smart cropping and resizing operations."""

from __future__ import annotations

import torch
from PIL import Image, ImageOps
from typing import Tuple, List


def compute_square_crop_bbox(bbox: torch.Tensor, img_width: int, img_height: int) -> torch.Tensor:
    """
    Compute square crop bbox centered on detection bbox.

    Returns unclamped coordinates (may extend outside image bounds).
    The square size equals max(bbox_width, bbox_height).
    """
    x1, y1, x2, y2 = bbox.int().tolist()

    # Clamp for size calculation only
    x1_c, y1_c = max(0, x1), max(0, y1)
    x2_c, y2_c = min(img_width, x2), min(img_height, y2)

    bbox_size = max(x2_c - x1_c, y2_c - y1_c)
    center_x = (x1_c + x2_c) // 2
    center_y = (y1_c + y2_c) // 2
    half_size = bbox_size // 2

    return torch.tensor([
        center_x - half_size,
        center_y - half_size,
        center_x + half_size,
        center_y + half_size
    ], dtype=torch.float32)


def smart_bbox_crop_with_mask(image: Image.Image, mask: torch.Tensor, bbox: torch.Tensor, target_size: int) -> Tuple[Image.Image, Image.Image, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Smart bounding box cropping. Returns cropped image, mask PIL, mask tensor, square_crop_bbox, offset."""
    img_width, img_height = image.size

    # Compute ideal square crop (unclamped)
    square_crop_bbox = compute_square_crop_bbox(bbox, img_width, img_height)
    sq_x1, sq_y1, sq_x2, sq_y2 = square_crop_bbox.int().tolist()

    # Clamp to image bounds for actual cropping
    crop_x1 = max(0, sq_x1)
    crop_y1 = max(0, sq_y1)
    crop_x2 = min(img_width, sq_x2)
    crop_y2 = min(img_height, sq_y2)

    # Crop image and mask
    cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    cropped_mask_tensor = mask[crop_y1:crop_y2, crop_x1:crop_x2].bool()

    # Convert mask to PIL for DINOv3
    mask_pil = Image.fromarray((mask.cpu().numpy() * 255).astype('uint8'), mode='L')
    cropped_mask_pil = mask_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Resize and pad to target size
    cropped_image = resize_and_pad_image(cropped_image, target_size)
    cropped_mask_pil = resize_and_pad_image(cropped_mask_pil, target_size)

    return cropped_image, cropped_mask_pil, cropped_mask_tensor, square_crop_bbox, (crop_x1, crop_y1)


def resize_and_pad_image(image: Image.Image, target_size: int) -> Image.Image:
    """Resize image preserving aspect ratio and pad to square."""
    original_width, original_height = image.size

    # Calculate scaling factor
    scale = min(target_size / original_width, target_size / original_height)

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize with high quality interpolation
    resized_image = image.resize((new_width, new_height), Image.BICUBIC)

    # Calculate padding
    pad_width = (target_size - new_width) // 2
    pad_height = (target_size - new_height) // 2

    # Add padding (black for images and masks)
    padded_image = ImageOps.expand(
        resized_image,
        (pad_width, pad_height, target_size - new_width - pad_width, target_size - new_height - pad_height),
        fill=0  # Black padding
    )

    return padded_image


def extract_cropped_detections(image: Image.Image, segmentation_output: dict, target_size: int = 512) -> List[Tuple[Image.Image, Image.Image, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]]:
    """Extract smart-cropped detection images and corresponding masks."""
    masks = segmentation_output["masks"]
    boxes = segmentation_output["boxes"]

    cropped_detections = []

    for mask, bbox in zip(masks, boxes):
        cropped_image, cropped_mask_pil, cropped_mask_tensor, square_crop_bbox, mask_offset = smart_bbox_crop_with_mask(image, mask, bbox, target_size)
        cropped_detections.append((cropped_image, cropped_mask_pil, bbox, square_crop_bbox, cropped_mask_tensor, mask_offset))

    return cropped_detections