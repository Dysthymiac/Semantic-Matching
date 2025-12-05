"""Standalone script for training PCA on preprocessed DINO features."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.config.config import MainConfig
from src.data.preprocessed_dataset import PreprocessedDataset
from src.pca.incremental_pca import IncrementalPCAProcessor


def train_pca_from_preprocessed_data(config: MainConfig) -> None:
    """Train PCA on all preprocessed detection features."""
    print(f"Training PCA from preprocessed data:")
    print(f"  Output root: {config.output_root}")
    print(f"  PCA components: {config.pca.n_components}")
    print(f"  PCA batch size: {config.pca.batch_size}")

    # Load preprocessed dataset
    preprocessed_dataset = PreprocessedDataset(config.output_root, batch_size=1000)
    total_detections = preprocessed_dataset.get_total_detection_count()

    if total_detections == 0:
        print("No preprocessed detections found! Run preprocess_dataset.py first.")
        return

    print(f"Found {total_detections} preprocessed detections")

    # Initialize PCA processor
    pca_processor = IncrementalPCAProcessor(config.pca, config.output_root)

    if pca_processor.is_fitted():
        print("PCA already fitted. Retraining from scratch...")
        # Clear existing PCA
        pca_processor = IncrementalPCAProcessor(config.pca, config.output_root)

    # Load all detections and train PCA in batches
    print("Loading detections and training PCA...")
    all_detections = preprocessed_dataset.iter_all_detections()

    # Process detections in batches for PCA training
    batch_size = config.pca.batch_size
    total_samples = 0

    for i in tqdm(range(0, len(all_detections), batch_size), desc="Training PCA batches"):
        batch_detections = all_detections[i:i + batch_size]

        success = pca_processor.fit_batch(batch_detections)
        if success:
            total_samples += len(batch_detections)

        # Print progress every 10 batches
        if (i // batch_size) % 10 == 0:
            pca_stats = pca_processor.get_stats_summary()
            print(f"  Processed {total_samples} detections, "
                  f"PCA samples: {pca_stats['n_samples_seen']}")

    # Print final statistics
    print(f"\nPCA training complete!")
    if pca_processor.is_fitted():
        pca_stats = pca_processor.get_stats_summary()
        print(f"Final PCA statistics:")
        print(f"  Total detections processed: {total_samples}")
        print(f"  PCA samples seen: {pca_stats['n_samples_seen']}")
        print(f"  PCA batches fitted: {pca_stats['n_batches_fitted']}")
        print(f"  Feature dimension: {pca_stats['feature_dim']}")
        print(f"  PCA components: {config.pca.n_components}")

        if 'total_variance_explained' in pca_stats:
            print(f"  Total variance explained: {pca_stats['total_variance_explained']:.3f}")
            print(f"  Top 10 components variance: {pca_stats['top_10_components_variance']:.3f}")

        # Get explained variance ratio for analysis
        variance_ratio = pca_processor.get_explained_variance_ratio()
        if variance_ratio is not None:
            print(f"  Variance per component (top 10): {variance_ratio[:10]}")

        print(f"\nPCA model saved to: {pca_processor.pca_dir}")
    else:
        print("PCA training failed - no valid features found!")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train PCA on preprocessed DINO features")
    parser.add_argument("config", type=Path, help="Path to configuration YAML file")

    args = parser.parse_args()

    # Load configuration
    config = MainConfig.from_yaml(args.config)

    # Train PCA
    train_pca_from_preprocessed_data(config)


if __name__ == "__main__":
    main()