"""Train GMM codebook for Fisher Vector encoding."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config.config import MainConfig
from src.data.preprocessed_dataset import PreprocessedDataset
from src.pca.incremental_pca import IncrementalPCAProcessor
from src.utils.sampling_utils import sample_patches_from_dataset
from src.utils.memory_monitor import force_garbage_collection, print_memory_summary
from src.codebook.gmm_trainer import (
    train_gmm,
    calculate_gmm_statistics,
    save_gmm_model
)


def main():
    """Main orchestrator for GMM training."""
    parser = argparse.ArgumentParser(description="Train GMM codebook from config")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = MainConfig.from_yaml(args.config)

    if not config.gmm:
        print("ERROR: No GMM configuration found in config file")
        return

    # Load dataset
    print(f"Loading dataset from: {config.output_root}")
    dataset = PreprocessedDataset(config.output_root)

    dataset_stats = {
        'total_detections': dataset.get_total_detection_count(),
        'total_images': dataset.get_processed_image_count(),
        'batch_count': dataset.get_batch_count()
    }
    print(f"Dataset: {dataset_stats['total_detections']:,} detections from {dataset_stats['total_images']:,} images")

    # Load PCA processor if configured
    pca_processor = None
    if config.gmm.use_pca:
        print("\nLoading PCA model...")
        pca_processor = IncrementalPCAProcessor(config.pca, config.output_root)

        if not pca_processor.is_fitted():
            print("ERROR: PCA not fitted. Run preprocessing first.")
            return

        pca_stats = pca_processor.get_stats_summary()
        print(f"PCA loaded ({pca_stats['n_samples_seen']:,} samples)")

    # Sample patches with on-the-fly PCA transformation
    print_memory_summary("Before sampling")
    sampled_patches, sampling_stats = sample_patches_from_dataset(
        dataset,
        n_samples=config.gmm.n_samples,
        random_seed=config.gmm.random_seed,
        pca_processor=pca_processor
    )

    # Clean up references to free memory
    del dataset
    if pca_processor:
        del pca_processor
    force_garbage_collection()
    print_memory_summary("After sampling and cleanup")

    # Train GMM
    print_memory_summary("Before GMM training")
    gmm = train_gmm(sampled_patches, config.gmm)
    print_memory_summary("After GMM training")

    # Calculate statistics
    gmm_stats = calculate_gmm_statistics(gmm, sampled_patches)

    # Prepare output path using auto-generated filename
    output_path = config.gmm_model_path

    # Prepare metadata
    metadata = {
        'dataset_stats': dataset_stats,
        'sampling_stats': sampling_stats,
        'gmm_stats': gmm_stats,
        'config': {
            'n_components': config.gmm.n_components,
            'covariance_type': config.gmm.covariance_type,
            'use_pca': config.gmm.use_pca,
            'pca_components': config.pca.n_components if config.gmm.use_pca else None
        },
        'config_path': str(args.config)
    }

    # Save model
    save_gmm_model(gmm, output_path, metadata)

    # Summary
    print("\n" + "="*50)
    print("GMM Training Complete!")
    print("="*50)
    print(f"  Components: {config.gmm.n_components}")
    print(f"  Dimensions: {sampled_patches.shape[1]}")
    print(f"  Samples: {sampled_patches.shape[0]:,}")
    print(f"  Converged: {gmm_stats['converged']}")
    print(f"  Model size: {gmm_stats['model_size_mb']:.2f} MB")
    if 'sample_log_likelihood' in gmm_stats:
        print(f"  Log-likelihood: {gmm_stats['sample_log_likelihood']:.2f}")


if __name__ == "__main__":
    main()