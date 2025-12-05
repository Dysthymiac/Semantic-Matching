"""SAM3 segmentation functionality using Transformers library."""

from __future__ import annotations

import torch
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor
from typing import List, Tuple

from ..config.config import SAMConfig
from ..utils.crop_utils import smart_bbox_crop_with_mask


class SAM3Segmenter:
    """Handles SAM3 image segmentation with species prompts using Transformers."""

    def __init__(self, config: SAMConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = Sam3Model.from_pretrained("facebook/sam3", force_download=False).to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", force_download=False)

    def segment_images_batch(self, images: list[Image.Image], species_list: list[str]) -> list[dict]:
        """Segment batch of images using their respective species-specific text prompts."""
        batch_size = len(images)

        # Prepare all prompts for all images
        all_prompts = []
        all_images = []
        prompt_to_image_map = []  # Track which image each prompt belongs to

        for img_idx, (image, species) in enumerate(zip(images, species_list)):
            prompts = self.config.species_prompts.get(species, [species])
            for prompt in prompts:
                all_prompts.append(prompt)
                all_images.append(image)
                prompt_to_image_map.append(img_idx)

        # Process entire batch at once
        inputs = self.processor(images=all_images, text=all_prompts, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get masks, boxes, scores for each prompt
        target_sizes = [img.size[::-1] for img in all_images]  # (height, width)
        results = self.processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)

        # Group results back by original image
        image_results = [{"masks": [], "boxes": [], "scores": []} for _ in range(batch_size)]

        for result, img_idx in zip(results, prompt_to_image_map):
            if len(result["masks"]) > 0:
                image_results[img_idx]["masks"].append(result["masks"])
                image_results[img_idx]["boxes"].append(result["boxes"])
                image_results[img_idx]["scores"].append(result["scores"])

        # Consolidate results for each image
        final_results = []
        for i, (image, result) in enumerate(zip(images, image_results)):
            if not result["masks"]:
                final_results.append({
                    "masks": torch.empty((0, *image.size[::-1])),
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,))
                })
            else:
                final_results.append({
                    "masks": torch.cat(result["masks"], dim=0),
                    "boxes": torch.cat(result["boxes"], dim=0),
                    "scores": torch.cat(result["scores"], dim=0)
                })

        return final_results

    def segment_image_with_species_prompt(self, image: Image.Image, species: str) -> dict:
        """Segment single image using species-specific text prompts - wrapper for batch method."""
        results = self.segment_images_batch([image], [species])
        return results[0]

    def filter_masks_by_score(self, segmentation_output: dict, min_score: float = 0.5) -> dict:
        """Filter masks by minimum confidence score, keeping all above threshold."""
        masks = segmentation_output["masks"]
        boxes = segmentation_output["boxes"]
        scores = segmentation_output["scores"]

        if scores.numel() == 0:
            return segmentation_output

        valid_indices = scores >= min_score

        return {
            "masks": masks[valid_indices],
            "boxes": boxes[valid_indices],
            "scores": scores[valid_indices]
        }