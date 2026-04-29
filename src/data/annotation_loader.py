"""Unified annotation loader for COCO and wildlife CSV formats."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..config.config import MainConfig
from .coco_loader import COCOAnnotation, COCOCategory, COCOImage, COCOLoader
from .wildlife_loader import WildlifeCSVConfig, WildlifeCSVLoader


class AnnotationLoader(Protocol):
    """Protocol for annotation loaders used across the pipeline."""

    @property
    def annotations(self) -> list[COCOAnnotation]:
        ...

    @property
    def images(self) -> dict[str, COCOImage]:
        ...

    @property
    def categories(self) -> dict[int, COCOCategory]:
        ...

    @property
    def viewpoints(self) -> set[str]:
        ...

    def get_image_path(self, image: COCOImage) -> Path:
        ...


def load_annotations(config: MainConfig) -> AnnotationLoader:
    """Create the appropriate annotation loader based on config."""
    if config.coco_json_path and config.wildlife_annotation_path:
        raise ValueError("Specify only one of coco_json_path or wildlife_annotation_path.")

    if config.wildlife_annotation_path:
        wildlife_config = WildlifeCSVConfig(
            annotations_path=config.wildlife_annotation_path,
            dataset_root=config.dataset_root,
            crop_to_bbox=True,
            split_filter=config.wildlife_split_filter,
            category_id=config.wildlife_category_id,
            species_name=config.wildlife_species_name,
        )
        return WildlifeCSVLoader(wildlife_config)

    if config.coco_json_path:
        return COCOLoader(config.coco_json_path, config.dataset_root)

    raise ValueError("No annotation path set. Provide coco_json_path or wildlife_annotation_path.")
