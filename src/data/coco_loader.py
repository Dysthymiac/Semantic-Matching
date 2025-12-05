"""COCO format annotation loader for viewpoint analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    def crop_image(self, image: Image.Image) -> Image.Image:
        """Crop image using bounding box coordinates."""
        return image.crop((self.x1, self.y1, self.x2, self.y2))

    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class COCOImage:
    """COCO image metadata."""
    uuid: str
    image_path: str
    width: int
    height: int
    latitude: float
    longitude: float
    datetime: str


@dataclass(frozen=True)
class COCOAnnotation:
    """Single COCO annotation with viewpoint label."""
    uuid: str
    image_uuid: str
    bbox: BoundingBox
    individual_id: str
    viewpoint: str
    category_id: int
    annot_census: bool
    annot_census_region: bool = False
    annot_manual: bool = False
    category: str = ""

    @property
    def image_id(self) -> str:
        """Get image UUID for compatibility."""
        return self.image_uuid


@dataclass(frozen=True)
class COCOCategory:
    """COCO category information."""
    id: int
    species: str


class COCOLoader:
    """Loads COCO format annotations and provides image access."""

    def __init__(self, coco_json_path: Path, dataset_root: Path | None = None, crop_to_bbox: bool = True, images_dir: Path | None = None) -> None:
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None

        coco_path = Path(coco_json_path)
        if coco_path.is_absolute():
            self.coco_json_path = coco_path
        else:
            self.coco_json_path = self.dataset_root / coco_path if self.dataset_root else coco_path

        if images_dir is not None:
            images_dir = Path(images_dir)
            if images_dir.is_absolute():
                self.images_dir = images_dir
            else:
                self.images_dir = self.dataset_root / images_dir
        else:
            self.images_dir = self.dataset_root

        self.crop_to_bbox = crop_to_bbox
        self._annotations: list[COCOAnnotation] = []
        self._images: dict[str, COCOImage] = {}
        self._categories: dict[int, COCOCategory] = {}
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Parse COCO JSON annotations."""
        with open(self.coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Load categories
        for cat_data in coco_data['categories']:
            category = COCOCategory(
                id=cat_data['id'],
                species=cat_data.get('species', cat_data.get('name', ''))
            )
            self._categories[category.id] = category

        # Load images
        for img_data in coco_data['images']:
            image = COCOImage(
                uuid=img_data['uuid'],
                image_path=img_data['image_path'],
                width=img_data['width'],
                height=img_data['height'],
                latitude=img_data['latitude'],
                longitude=img_data['longitude'],
                datetime=img_data['datetime']
            )
            self._images[image.uuid] = image

        # Load annotations
        for ann_data in coco_data['annotations']:
            # Handle different bbox formats
            if 'bbox' in ann_data:
                bbox = BoundingBox(*ann_data['bbox'])
            elif all(k in ann_data for k in ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']):
                x, y, w, h = ann_data['bbox_x'], ann_data['bbox_y'], ann_data['bbox_w'], ann_data['bbox_h']
                bbox = BoundingBox(x, y, x + w, y + h)
            else:
                bbox = BoundingBox(0, 0, 0, 0)

            viewpoint = ann_data.get('viewpoint', 'unknown')
            if not isinstance(viewpoint, str):
                viewpoint = "unknown"

            annotation = COCOAnnotation(
                uuid=ann_data['uuid'],
                image_uuid=ann_data['image_uuid'],
                bbox=bbox,
                viewpoint=viewpoint,
                individual_id=ann_data.get('individual_id', ann_data.get('name_uuid', '')),
                category_id=ann_data['category_id'],
                annot_census=ann_data.get('annot_census', False),
                annot_census_region=ann_data.get('annot_census_region', False),
                annot_manual=ann_data.get('annot_manual', False),
                category=ann_data.get('category', '')
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
        image_path = Path(image.image_path)

        if image_path.is_absolute():
            return image_path
        else:
            if self.images_dir:
                return self.images_dir / image_path
            else:
                return image_path

    def get_image_path_from_annotation(self, annotation: COCOAnnotation) -> Path:
        """Get full path to image file from annotation."""
        return self.get_image_path(self._images[annotation.image_uuid])

    def load_cropped_image(self, annotation: COCOAnnotation) -> Image.Image:
        """Load and crop image according to annotation."""
        image_path = self.get_image_path_from_annotation(annotation)
        image = Image.open(image_path).convert('RGB')
        return annotation.bbox.crop_image(image)

    def load_full_image(self, annotation: COCOAnnotation) -> Image.Image:
        """Load full image without cropping."""
        image_path = self.get_image_path_from_annotation(annotation)
        return Image.open(image_path).convert('RGB')

    def load_image(self, annotation: COCOAnnotation) -> Image.Image:
        """Load image according to crop_to_bbox setting."""
        if self.crop_to_bbox:
            return self.load_cropped_image(annotation)
        else:
            return self.load_full_image(annotation)

    def filter_by_category_id(self, category_ids: list[int]) -> list[COCOAnnotation]:
        """Filter annotations by category IDs."""
        return [ann for ann in self._annotations if ann.category_id in category_ids]

    def __len__(self) -> int:
        return len(self._annotations)