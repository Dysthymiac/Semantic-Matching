"""Wildlife CSV annotation loader (non-COCO format)."""

from __future__ import annotations

import csv
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from .coco_loader import BoundingBox, COCOAnnotation, COCOCategory, COCOImage


@dataclass(frozen=True)
class WildlifeCSVConfig:
    """Configuration for wildlife CSV annotations."""
    annotations_path: Path
    dataset_root: Path | None = None
    images_dir: Path | None = None
    crop_to_bbox: bool = True
    split_filter: List[str] | None = None
    category_id: int = 1
    species_name: str = "wildlife"


class WildlifeCSVLoader:
    """Loads wildlife CSV annotations and provides COCO-like accessors."""

    def __init__(self, config: WildlifeCSVConfig) -> None:
        self.dataset_root = Path(config.dataset_root) if config.dataset_root is not None else None

        csv_path = Path(config.annotations_path)
        if csv_path.is_absolute():
            self.annotations_path = csv_path
        else:
            self.annotations_path = self.dataset_root / csv_path if self.dataset_root else csv_path

        if config.images_dir is not None:
            images_dir = Path(config.images_dir)
            if images_dir.is_absolute():
                self.images_dir = images_dir
            else:
                self.images_dir = self.dataset_root / images_dir if self.dataset_root else images_dir
        else:
            self.images_dir = self.dataset_root

        self.crop_to_bbox = config.crop_to_bbox
        self.split_filter = config.split_filter
        self.category_id = config.category_id
        self.species_name = config.species_name

        self._annotations: list[COCOAnnotation] = []
        self._images: dict[str, COCOImage] = {}
        self._categories: dict[int, COCOCategory] = {}

        self._load_annotations()

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        if self.images_dir:
            return self.images_dir / path
        return path

    def _parse_bbox(self, raw_bbox: str) -> BoundingBox:
        try:
            parsed = literal_eval(raw_bbox)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
            x, y, w, h = parsed
            return BoundingBox(float(x), float(y), float(x) + float(w), float(y) + float(h))

        return BoundingBox(0.0, 0.0, 0.0, 0.0)

    def _get_image_size(self, image_path: str) -> tuple[int, int]:
        full_path = self._resolve_image_path(image_path)
        with Image.open(full_path) as image:
            width, height = image.size
        return width, height

    def _load_annotations(self) -> None:
        self._categories[self.category_id] = COCOCategory(
            id=self.category_id,
            species=self.species_name,
        )

        with open(self.annotations_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if self.split_filter:
                    split = (row.get("split") or "").strip()
                    if split not in self.split_filter:
                        continue

                image_path = (row.get("path") or "").strip()
                if not image_path:
                    continue

                image_uuid = Path(image_path).stem
                if image_uuid not in self._images:
                    width, height = self._get_image_size(image_path)
                    self._images[image_uuid] = COCOImage(
                        uuid=image_uuid,
                        image_path=image_path,
                        width=width,
                        height=height,
                        latitude=0.0,
                        longitude=0.0,
                        datetime="",
                    )

                bbox = self._parse_bbox(row.get("bbox", ""))

                annotation_uuid = (row.get("annotation_uuid") or "").strip()
                if not annotation_uuid:
                    annotation_uuid = f"{image_uuid}_{idx:06d}"

                viewpoint = row.get("viewpoint", "unknown")
                if not isinstance(viewpoint, str):
                    viewpoint = "unknown"

                individual_id = row.get("identity", "")
                if individual_id is not None and not isinstance(individual_id, str):
                    individual_id = str(individual_id)

                annotation = COCOAnnotation(
                    uuid=annotation_uuid,
                    image_uuid=image_uuid,
                    bbox=bbox,
                    viewpoint=viewpoint,
                    individual_id=individual_id or "",
                    category_id=self.category_id,
                    annot_census=False,
                    annot_census_region=False,
                    annot_manual=False,
                    category=self.species_name,
                )
                self._annotations.append(annotation)

    @property
    def annotations(self) -> list[COCOAnnotation]:
        """Get all annotations."""
        return self._annotations.copy()

    @property
    def images(self) -> dict[str, COCOImage]:
        """Get all images indexed by UUID."""
        return self._images.copy()

    @property
    def categories(self) -> dict[int, COCOCategory]:
        """Get all categories indexed by ID."""
        return self._categories.copy()

    @property
    def viewpoints(self) -> set[str]:
        """Get unique viewpoint labels."""
        return {ann.viewpoint for ann in self._annotations}

    def get_image_path(self, image: COCOImage) -> Path:
        """Get full path to image file."""
        return self._resolve_image_path(image.image_path)

    def get_image_path_from_annotation(self, annotation: COCOAnnotation) -> Path:
        """Get full path to image file from annotation."""
        return self.get_image_path(self._images[annotation.image_uuid])

    def load_cropped_image(self, annotation: COCOAnnotation) -> Image.Image:
        """Load and crop image according to annotation."""
        image_path = self.get_image_path_from_annotation(annotation)
        image = Image.open(image_path).convert("RGB")
        return annotation.bbox.crop_image(image)

    def load_full_image(self, annotation: COCOAnnotation) -> Image.Image:
        """Load full image without cropping."""
        image_path = self.get_image_path_from_annotation(annotation)
        return Image.open(image_path).convert("RGB")

    def load_image(self, annotation: COCOAnnotation) -> Image.Image:
        """Load image according to crop_to_bbox setting."""
        if self.crop_to_bbox:
            return self.load_cropped_image(annotation)
        return self.load_full_image(annotation)

    def filter_by_category_id(self, category_ids: list[int]) -> list[COCOAnnotation]:
        """Filter annotations by category IDs."""
        return [ann for ann in self._annotations if ann.category_id in category_ids]

    def __len__(self) -> int:
        return len(self._annotations)
