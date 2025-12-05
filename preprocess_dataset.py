"""Main preprocessing script for SAM3 segmentation and DINOv3 feature extraction on images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from src.config.config import MainConfig
from src.data.coco_loader import COCOLoader, COCOImage
from src.data.preprocessed_dataset import PreprocessedDataset
from src.pipeline.preprocessing_pipeline import PreprocessingPipeline
from src.pca.incremental_pca import IncrementalPCAProcessor
from src.utils.memory_monitor import get_memory_stats, print_memory_summary, force_garbage_collection
import pickle


def get_all_image_paths(coco_loader: COCOLoader) -> List[str]:
    """Get all unique image paths from COCO dataset."""
    image_paths = []
    for image in coco_loader.images.values():
        full_path = str(coco_loader.get_image_path(image))
        image_paths.append(full_path)
    # Remove duplicates while preserving deterministic order
    seen = set()
    unique_paths = []
    for path in image_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    unique_paths.sort()  # Ensure consistent ordering across runs
    return unique_paths


def run_preprocessing(config: MainConfig) -> None:
    """Main preprocessing function for image-based processing."""
    print(f"Starting image-based preprocessing with config:")
    print(f"  COCO JSON: {config.coco_json_path}")
    print(f"  Dataset root: {config.dataset_root}")
    print(f"  Output root: {config.output_root}")

    # Load COCO dataset to get image paths (ignore annotations for unsupervised approach)
    coco_loader = COCOLoader(config.coco_json_path, config.dataset_root)
    print(f"Loaded COCO dataset with {len(coco_loader.images)} images")

    # Get all species from config
    species_list = list(config.sam.species_prompts.keys())
    print(f"Will search for species: {species_list}")

    # Get all image paths
    all_image_paths = get_all_image_paths(coco_loader)
    print(f"Found {len(all_image_paths)} unique images")

    # Initialize preprocessed dataset and PCA processor
    preprocessed_dataset = PreprocessedDataset(config.output_root, batch_size=300)
    pca_processor = IncrementalPCAProcessor(config.pca, config.output_root)

    # Get unprocessed images
    unprocessed_image_paths = preprocessed_dataset.get_unprocessed_images(all_image_paths)
    print(f"Found {len(unprocessed_image_paths)} unprocessed images")
    print(f"Already processed: {preprocessed_dataset.get_processed_image_count()} images")

    # Print PCA status
    if pca_processor.is_fitted():
        pca_stats = pca_processor.get_stats_summary()
        print(f"PCA already fitted with {pca_stats['n_samples_seen']} samples")
    else:
        print("PCA not yet fitted - will start incremental fitting")

    if not unprocessed_image_paths:
        print("All images already processed!")
        return

    # Process images
    total_detections = 0
    failed_count = 0

    batch_size = config.processing_batch_size
    print(f"Starting batch image processing (batch size: {batch_size})...")
    pca_fit_batch = []  # Accumulate detections for PCA fitting

    # Create preprocessing pipeline (models initialized once)
    pipeline = PreprocessingPipeline(config)

    # Initial memory baseline
    print_memory_summary("BASELINE")

    # Process in batches
    for i in tqdm(range(0, len(unprocessed_image_paths), batch_size), desc="Processing image batches"):
        batch_num = i // batch_size + 1

        # Memory check at start of each batch
        print_memory_summary(f"BATCH {batch_num} START")

        batch_paths = unprocessed_image_paths[i:i + batch_size]

        # Load images for this batch
        batch_images = []
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            batch_images.append(image)

        # Process entire batch
        batch_detection_results = pipeline.process_images_batch(batch_images, batch_paths, species_list)

        # Clear PIL images immediately to free memory
        batch_images.clear()
        del batch_images

        # Collect ALL detections from processing batch
        all_detections_from_batch = []
        batch_has_detections = False

        for path, detections in zip(batch_paths, batch_detection_results):
            detection_count = len(detections)

            if detection_count > 0:
                total_detections += detection_count
                batch_has_detections = True
                all_detections_from_batch.extend(detections)

                # Collect detections for PCA fitting if iterative fitting is enabled
                if config.pca.iterative_fitting:
                    pca_fit_batch.extend(detections)
            else:
                failed_count += 1
                print(f"No detections found for image: {path}")

        # Save ALL detections from processing batch at ONCE (not per image)
        if all_detections_from_batch:
            preprocessed_dataset.add_detections_for_batch(all_detections_from_batch)

            # Check for corrupted batch and clean up PCA training history
            corrupted_images = preprocessed_dataset.get_and_clear_corrupted_images()
            if corrupted_images:
                pca_processor.remove_corrupted_images_from_training(corrupted_images)

        # Clear detection collections immediately
        all_detections_from_batch.clear()
        del all_detections_from_batch

        # Clear batch detection results to free memory
        batch_detection_results.clear()
        del batch_detection_results

        # Force garbage collection and GPU cache clearing
        force_garbage_collection()

        # Fit PCA on every processing batch if iterative fitting is enabled
        if config.pca.iterative_fitting and pca_fit_batch:
            print(f"Fitting PCA on batch of {len(pca_fit_batch)} detections...")
            success = pca_processor.fit_batch(pca_fit_batch)
            if success:
                pca_stats = pca_processor.get_stats_summary()
                print(f"PCA updated - total samples: {pca_stats['n_samples_seen']}")
            pca_fit_batch.clear()  # Clear batch after fitting

        # Memory check after batch completion
        print_memory_summary(f"BATCH {batch_num} END")

        # Print progress every 100 images
        processed_count = preprocessed_dataset.get_processed_image_count()
        if processed_count % 100 == 0:
            print(f"Processed: {processed_count}, "
                  f"Total detections: {total_detections}, "
                  f"Failed: {failed_count}")

    # Final PCA fitting on remaining batch
    if pca_fit_batch and config.pca.iterative_fitting:
        print(f"Final PCA fitting on remaining {len(pca_fit_batch)} detections...")
        pca_processor.fit_batch(pca_fit_batch)

    print(f"\nPreprocessing complete!")
    print(f"Final dataset stats:")
    print(f"  Processed images: {preprocessed_dataset.get_processed_image_count()}")
    print(f"  Total detections: {preprocessed_dataset.get_total_detection_count()}")
    print(f"  Batch files: {preprocessed_dataset.get_batch_count()}")
    print(f"  Failed images: {failed_count}")

    # Print final PCA statistics
    if pca_processor.is_fitted():
        pca_stats = pca_processor.get_stats_summary()
        print(f"  PCA Statistics:")
        print(f"    Samples seen: {pca_stats['n_samples_seen']}")
        print(f"    Batches fitted: {pca_stats['n_batches_fitted']}")
        print(f"    Components: {config.pca.n_components}")
        if 'total_variance_explained' in pca_stats:
            print(f"    Total variance explained: {pca_stats['total_variance_explained']:.3f}")
    else:
        print(f"  PCA not fitted (no valid features found)")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Preprocess images with SAM3 and DINOv3")
    parser.add_argument("config", type=Path, help="Path to configuration YAML file")

    args = parser.parse_args()

    # Load configuration
    config = MainConfig.from_yaml(args.config)

    # Run preprocessing
    run_preprocessing(config)


if __name__ == "__main__":
    main()