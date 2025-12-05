"""Compute Fisher Vectors for all detections in the dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from tqdm import tqdm

import numpy as np
from sklearn.decomposition import IncrementalPCA

from src.config.config import MainConfig
from src.data.preprocessed_dataset import PreprocessedDataset
from src.data.fv_dataset import FisherVectorDataset
from src.features.fisher_vector import encode_detection_fisher_vector, compute_fisher_vector_stats
from src.pca.incremental_pca import IncrementalPCAProcessor
from src.utils.batch_storage import load_batch_from_file
from src.utils.memory_monitor import force_garbage_collection, print_memory_summary
from src.codebook.gmm_trainer import load_gmm_model


def main():
    """Main Fisher Vector computation pipeline."""
    parser = argparse.ArgumentParser(description="Compute Fisher Vectors from preprocessed features")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = MainConfig.from_yaml(args.config)

    if not config.fisher_vector:
        print("ERROR: No Fisher Vector configuration found in config file")
        print("Add 'fisher_vector' section to your config")
        return

    if not config.gmm:
        print("ERROR: No GMM configuration found. GMM config is required for Fisher Vectors")
        return

    # Load GMM model - path is auto-generated from GMM config
    gmm_path = config.gmm_model_path
    if not gmm_path.exists():
        print(f"ERROR: GMM model not found at {gmm_path}")
        print("Run train_gmm.py first to train the GMM")
        return

    print(f"Loading GMM from: {gmm_path}")
    gmm, gmm_metadata = load_gmm_model(gmm_path)
    print(f"GMM loaded: {gmm.n_components} components, {gmm.covariance_type} covariance")

    # Load PCA processor if needed
    pca_processor = None
    if config.fisher_vector.use_pca:
        print("Loading PCA processor for feature transformation...")
        pca_processor = IncrementalPCAProcessor(config.pca, config.output_root)
        if not pca_processor.is_fitted():
            print("ERROR: PCA not fitted. Run preprocessing pipeline first.")
            return
        print(f"PCA loaded: {pca_processor.stats['n_samples_seen']:,} samples")

    # Load preprocessed dataset
    print(f"Loading dataset from: {config.output_root}")
    dataset = PreprocessedDataset(config.output_root)
    print(f"Dataset: {dataset.get_total_detection_count():,} detections")

    # Initialize Fisher Vector datasets (separate for original and reduced)
    fv_dataset_original = FisherVectorDataset(config.output_root / "fisher_vectors_original")
    fv_dataset_reduced = FisherVectorDataset(config.output_root / "fisher_vectors_reduced")

    # Initialize PCA for Fisher Vectors
    fv_pca = IncrementalPCA(
        n_components=config.fisher_vector.fv_pca_components,
        batch_size=config.fisher_vector.batch_size
    )

    # Process batches
    batch_files = sorted(dataset._index['batch_to_detections'].keys())
    print(f"\n=== PASS 1: Computing Fisher Vectors and fitting PCA ===")
    print(f"Processing {len(batch_files)} batch files...")

    total_processed = 0
    fv_stats_accumulator = []
    accumulated_fvs = []  # Accumulate FVs until we have enough for PCA

    print_memory_summary("Start Pass 1")

    # Pass 1: Compute FVs, save originals, fit incremental PCA
    for batch_idx, batch_file in enumerate(tqdm(batch_files, desc="Pass 1: Computing FVs")):
        batch_path = dataset.output_root / batch_file

        # Load batch
        batch_data = load_batch_from_file(batch_path)
        detection_ids = dataset._index['batch_to_detections'][batch_file]

        # Process all detections in this batch
        batch_fvs = {}
        for det_id in detection_ids:
            if det_id not in batch_data:
                continue

            detection = batch_data[det_id]

            # Encode Fisher Vector
            fv = encode_detection_fisher_vector(detection, gmm, pca_processor)

            if fv is not None:
                batch_fvs[det_id] = fv
                fv_stats_accumulator.append(compute_fisher_vector_stats(fv))
                total_processed += 1

        # Save original Fisher Vectors for this batch
        if batch_fvs:
            fv_dataset_original.save_batch(
                batch_fvs,
                batch_idx,
                metadata={
                    'gmm_components': gmm.n_components,
                    'pca_components': config.pca.n_components if config.fisher_vector.use_pca else None,
                    'fv_dimension': next(iter(batch_fvs.values())).shape[0]
                }
            )

            # Accumulate FVs for PCA fitting
            fv_array = np.vstack(list(batch_fvs.values()))
            accumulated_fvs.append(fv_array)

            # Check if we have enough samples for PCA
            total_accumulated = sum(arr.shape[0] for arr in accumulated_fvs)

            if total_accumulated >= config.fisher_vector.fv_pca_components:
                # We have enough samples, fit PCA
                combined_fvs = np.vstack(accumulated_fvs)

                if batch_idx == 0 or batch_idx == 1:
                    print(f"\nFisher Vector dimension: {combined_fvs.shape[1]:,}")
                    print(f"Fitting PCA with {combined_fvs.shape[0]} FVs (batch {batch_idx+1})")

                fv_pca.partial_fit(combined_fvs)
                accumulated_fvs = []  # Clear accumulator after fitting

        # Clean up memory
        del batch_data
        force_garbage_collection()

    # Fit any remaining accumulated FVs
    if accumulated_fvs:
        combined_fvs = np.vstack(accumulated_fvs)
        print(f"Final PCA fit with {combined_fvs.shape[0]} remaining FVs")
        fv_pca.partial_fit(combined_fvs)
        accumulated_fvs = []

    # Save FV PCA model
    fv_pca_path = config.output_root / "fisher_vectors" / "fv_pca.pkl"
    fv_pca_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fv_pca_path, 'wb') as f:
        pickle.dump(fv_pca, f)
    print(f"\nPCA model saved to: {fv_pca_path}")
    print(f"PCA fitted on {total_processed:,} Fisher Vectors")

    # Print Pass 1 statistics
    if fv_stats_accumulator:
        avg_stats = {}
        for key in fv_stats_accumulator[0].keys():
            if key != 'dim':
                values = [s[key] for s in fv_stats_accumulator]
                avg_stats[key] = np.mean(values)

        print(f"\nOriginal Fisher Vector Statistics:")
        print(f"  Dimension: {fv_stats_accumulator[0]['dim']:,}")
        print(f"  Average L2 norm: {avg_stats['norm']:.2f}")
        print(f"  Average sparsity: {avg_stats['sparsity']:.2%}")

    print_memory_summary("End Pass 1")

    # Pass 2: Load original FVs, transform with PCA, save reduced versions
    print(f"\n=== PASS 2: Transforming Fisher Vectors with PCA ===")
    print(f"Reducing from {fv_stats_accumulator[0]['dim']:,} to {config.fisher_vector.fv_pca_components} dimensions")

    for batch_idx in tqdm(range(len(batch_files)), desc="Pass 2: Transforming FVs"):
        # Load original Fisher Vectors
        batch_file = f"batch_{batch_idx:03d}.pkl"
        original_fvs = fv_dataset_original.load_batch(batch_file)

        if original_fvs:
            # Transform with PCA
            transformed_fvs = {}
            for det_id, fv in original_fvs.items():
                fv_reduced = fv_pca.transform(fv.reshape(1, -1))[0].astype(np.float32)
                transformed_fvs[det_id] = fv_reduced

            # Save reduced Fisher Vectors
            fv_dataset_reduced.save_batch(
                transformed_fvs,
                batch_idx,
                metadata={
                    'gmm_components': gmm.n_components,
                    'pca_components': config.pca.n_components if config.fisher_vector.use_pca else None,
                    'fv_pca_components': config.fisher_vector.fv_pca_components,
                    'original_fv_dimension': fv_stats_accumulator[0]['dim'] if fv_stats_accumulator else None
                }
            )

    print("\n" + "="*50)
    print("Fisher Vector Processing Complete!")
    print("="*50)
    print(f"Total detections processed: {total_processed:,}")
    print(f"Original FVs saved to: {fv_dataset_original.fv_root}")
    print(f"Reduced FVs saved to: {fv_dataset_reduced.fv_root}")
    print(f"Dimension reduction: {fv_stats_accumulator[0]['dim']:,} → {config.fisher_vector.fv_pca_components}")

    print_memory_summary("End")


if __name__ == "__main__":
    main()