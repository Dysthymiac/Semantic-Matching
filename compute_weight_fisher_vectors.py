"""Compute weight-augmented Fisher Vectors (2KD+K) for all detections.

Uses the theoretically correct Fisher score with three gradient types:
  - 0th order: gradient w.r.t. mixing weights (scalar per component, K total)
  - 1st order: gradient w.r.t. means (D-dim per component, KD total)
  - 2nd order: gradient w.r.t. variances (D-dim per component, KD total)

Total descriptor dimensionality per detection: K * (2D + 1).

Outputs:
  - weight_fisher_vectors_original/: Power+L2 normalized FVs (FisherVectorDataset)
  - weight_fisher_vectors_reduced/: PCA-reduced FVs (FisherVectorDataset)
  - weight_fisher_vectors_raw.pkl: Raw (unnormalized) FVs for block masking experiments

Usage:
    python compute_weight_fisher_vectors.py --config config_zebra_test.yaml
"""

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
from src.features.fisher_vector import compute_fisher_vector_stats
from src.pca.incremental_pca import IncrementalPCAProcessor
from src.codebook.gmm_trainer import load_gmm_model
from src.utils.batch_storage import load_batch_from_file
from src.utils.memory_monitor import force_garbage_collection, print_memory_summary


def compute_weight_augmented_fv_raw(features: np.ndarray, gmm) -> np.ndarray:
    """Compute raw (unnormalized) weight-augmented Fisher vector.

    Per component k, the descriptor has:
      - Weight grad (scalar):  (1/sqrt(w_k)) * (N_k/N - w_k)
      - Mean grad (D-dim):     (1/sqrt(N_k)) * sum_i gamma_k(x_i) * (x_i - mu_k) / sigma_k
      - Var grad (D-dim):      (1/sqrt(2*N_k)) * sum_i gamma_k(x_i) * ((x_i-mu_k)^2/sigma_k^2 - 1)

    Layout per component: [weight(1), mean(D), var(D)] -> total 2KD + K.
    Returns raw vector WITHOUT power or L2 normalization.

    Args:
        features: [N, D] PCA-transformed patch features.
        gmm: Trained GaussianMixture (diagonal covariance).

    Returns:
        Raw Fisher vector of shape [K*(2*D+1)], float32.
    """
    N, D = features.shape
    K = gmm.n_components
    block = 2 * D + 1  # per-component block size

    gamma = gmm.predict_proba(features)  # [N, K]
    N_k = gamma.sum(axis=0)  # [K]
    w_k = gmm.weights_  # [K]

    means = gmm.means_  # [K, D]
    if gmm.covariance_type == 'diag':
        variances = gmm.covariances_  # [K, D]
    elif gmm.covariance_type == 'full':
        variances = np.array([np.diag(cov) for cov in gmm.covariances_])
    else:
        raise ValueError(f"Unsupported covariance type: {gmm.covariance_type}")

    stds = np.sqrt(variances)  # [K, D]

    fv = np.zeros(K * block, dtype=np.float64)

    for k in range(K):
        offset = k * block

        if N_k[k] < 1e-10:
            continue

        # 0th order: weight term (scalar)
        fv[offset] = np.sqrt(N_k[k] / N)

        gamma_k = gamma[:, k]
        diff = (features - means[k]) / stds[k]

        # 1st order: mean gradient (D-dim)
        g_mu = (gamma_k[:, None] * diff).sum(axis=0) / np.sqrt(N_k[k])
        # 2nd order: variance gradient (D-dim)
        g_sigma = (gamma_k[:, None] * (diff ** 2 - 1)).sum(axis=0) / np.sqrt(2.0 * N_k[k])

        fv[offset + 1:offset + 1 + D] = g_mu
        fv[offset + 1 + D:offset + block] = g_sigma

    return fv.astype(np.float32)


def normalize_fv(raw_fv: np.ndarray) -> np.ndarray:
    """Power normalization + L2 normalization for a single FV."""
    fv = raw_fv.copy()
    signs = np.sign(fv)
    np.abs(fv, out=fv)
    np.sqrt(fv, out=fv)
    fv *= signs
    norm = np.linalg.norm(fv)
    if norm > 1e-10:
        fv /= norm
    return fv.astype(np.float32)


def main():
    """Main weight-augmented Fisher Vector computation pipeline."""
    parser = argparse.ArgumentParser(description="Compute weight-augmented Fisher Vectors (2KD+K)")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = MainConfig.from_yaml(args.config)

    if not config.fisher_vector:
        print("ERROR: No Fisher Vector configuration found in config file")
        return

    if not config.gmm:
        print("ERROR: No GMM configuration found")
        return

    # Load GMM
    gmm_path = config.gmm_model_path
    if not gmm_path.exists():
        print(f"ERROR: GMM model not found at {gmm_path}")
        print("Run train_gmm.py first to train the GMM")
        return

    print(f"Loading GMM from: {gmm_path}")
    gmm, gmm_metadata = load_gmm_model(gmm_path)
    K, D = gmm.means_.shape
    fv_dim = K * (2 * D + 1)
    print(f"GMM: {K} components, {D} dims -> FV dim: {fv_dim} (2KD+K)")

    # Load PCA processor
    pca_processor = None
    if config.fisher_vector.use_pca:
        print("Loading PCA processor for feature transformation...")
        pca_processor = IncrementalPCAProcessor(config.pca, config.output_root)

        if config.pca.full_pca_path:
            print(f"Loading full PCA from: {config.pca.full_pca_path}")
            pca_processor.load_full_pca(Path(config.pca.full_pca_path))
            print(f"Using PCA band: components [{config.pca.start_component}:{config.pca.end_component}]")
        elif not pca_processor.is_fitted():
            print("ERROR: PCA not fitted. Run preprocessing pipeline first or set pca.full_pca_path.")
            return
        else:
            print(f"PCA loaded: {pca_processor.stats['n_samples_seen']:,} samples")

    # Load dataset
    print(f"Loading dataset from: {config.output_root}")
    dataset = PreprocessedDataset(config.output_root)
    print(f"Dataset: {dataset.get_total_detection_count():,} detections")

    # Initialize output datasets
    fv_dataset_original = FisherVectorDataset(config.output_root / "weight_fisher_vectors_original")
    fv_dataset_reduced = FisherVectorDataset(config.output_root / "weight_fisher_vectors_reduced")

    # Initialize PCA for Fisher Vectors
    fv_pca = IncrementalPCA(
        n_components=config.fisher_vector.fv_pca_components,
        batch_size=config.fisher_vector.batch_size
    )

    # Process batches
    batch_files = sorted(dataset._index['batch_to_detections'].keys())

    # ========== PASS 1: Compute raw FVs, normalize, save originals, fit PCA ==========
    print(f"\n{'='*60}")
    print("PASS 1: Computing weight-augmented Fisher Vectors (2KD+K)")
    print(f"{'='*60}")

    total_processed = 0
    fv_stats_accumulator = []
    accumulated_fvs = []

    # Also accumulate raw FVs for bulk pickle (notebook experiments)
    all_det_ids_raw = []
    all_fvs_raw_list = []

    print_memory_summary("Start Pass 1")

    for batch_idx, batch_file in enumerate(tqdm(batch_files, desc="Pass 1: Computing FVs")):
        batch_path = dataset.output_root / batch_file
        batch_data = load_batch_from_file(batch_path)
        detection_ids = dataset._index['batch_to_detections'][batch_file]

        batch_fvs = {}
        for det_id in detection_ids:
            if det_id not in batch_data:
                continue

            detection = batch_data[det_id]

            # Extract valid patches
            features = detection.features  # [D_raw, H, W]
            mask = detection.patch_mask     # [H, W]
            D_raw, H, W = features.shape

            features_flat = features.reshape(D_raw, -1).T  # [H*W, D_raw]
            mask_flat = mask.flatten().bool()
            valid_features = features_flat[mask_flat].cpu().numpy()

            if valid_features.shape[0] == 0:
                continue

            # Apply PCA
            if pca_processor is not None:
                valid_features = pca_processor.transform_with_band(valid_features)

            # Compute raw weight-augmented FV
            raw_fv = compute_weight_augmented_fv_raw(valid_features, gmm)

            # Save raw for bulk pickle
            all_det_ids_raw.append(det_id)
            all_fvs_raw_list.append(raw_fv)

            # Normalize (power + L2) for original dataset
            fv_normalized = normalize_fv(raw_fv)

            batch_fvs[det_id] = fv_normalized
            fv_stats_accumulator.append(compute_fisher_vector_stats(fv_normalized))
            total_processed += 1

        # Save normalized Fisher Vectors (original)
        if batch_fvs:
            fv_dataset_original.save_batch(
                batch_fvs,
                batch_idx,
                metadata={
                    'type': 'weight_fisher',
                    'gmm_components': K,
                    'pca_components': config.pca.n_components if config.fisher_vector.use_pca else None,
                    'fv_dimension': fv_dim,
                    'formulation': '2KD+K (weight + mean + variance gradients)',
                }
            )

            # Accumulate for PCA fitting
            fv_array = np.vstack(list(batch_fvs.values()))
            accumulated_fvs.append(fv_array)

            total_accumulated = sum(arr.shape[0] for arr in accumulated_fvs)
            if total_accumulated >= config.fisher_vector.fv_pca_components:
                combined_fvs = np.vstack(accumulated_fvs)

                if batch_idx <= 1:
                    print(f"\nFisher Vector dimension: {combined_fvs.shape[1]:,}")
                    print(f"Fitting PCA with {combined_fvs.shape[0]} FVs (batch {batch_idx+1})")

                fv_pca.partial_fit(combined_fvs)
                accumulated_fvs = []

        del batch_data
        force_garbage_collection()

    # Fit remaining accumulated FVs
    if accumulated_fvs:
        combined_fvs = np.vstack(accumulated_fvs)
        print(f"Final PCA fit with {combined_fvs.shape[0]} remaining FVs")
        fv_pca.partial_fit(combined_fvs)
        accumulated_fvs = []

    # Save FV PCA model
    fv_pca_path = config.output_root / "weight_fisher_vectors" / "fv_pca.pkl"
    fv_pca_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fv_pca_path, 'wb') as f:
        pickle.dump(fv_pca, f)
    print(f"\nPCA model saved to: {fv_pca_path}")
    print(f"PCA fitted on {total_processed:,} Fisher Vectors")

    # Save raw FVs bulk pickle for notebook experiments
    raw_pkl_path = config.output_root / "weight_fisher_vectors_raw.pkl"
    print(f"\nSaving raw (unnormalized) FVs to: {raw_pkl_path}")
    all_fvs_raw = np.vstack(all_fvs_raw_list)
    del all_fvs_raw_list
    with open(raw_pkl_path, 'wb') as f:
        pickle.dump({"det_ids": all_det_ids_raw, "fvs_raw": all_fvs_raw}, f,
                     protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(all_det_ids_raw)} raw FVs, shape: {all_fvs_raw.shape}, "
          f"size: {all_fvs_raw.nbytes / 1e6:.1f} MB")
    del all_fvs_raw, all_det_ids_raw

    # Print Pass 1 statistics
    if fv_stats_accumulator:
        avg_stats = {}
        for key in fv_stats_accumulator[0].keys():
            if key != 'dim':
                values = [s[key] for s in fv_stats_accumulator]
                avg_stats[key] = np.mean(values)

        print(f"\nNormalized Fisher Vector Statistics:")
        print(f"  Dimension: {fv_stats_accumulator[0]['dim']:,} (2KD+K = 2*{K}*{D}+{K})")
        print(f"  Average L2 norm: {avg_stats['norm']:.4f}")
        print(f"  Average sparsity: {avg_stats['sparsity']:.2%}")

    print_memory_summary("End Pass 1")

    # ========== PASS 2: Transform with PCA ==========
    print(f"\n{'='*60}")
    print("PASS 2: Transforming Fisher Vectors with PCA")
    print(f"Reducing from {fv_stats_accumulator[0]['dim']:,} to {config.fisher_vector.fv_pca_components} dimensions")
    print(f"{'='*60}")

    for batch_idx in tqdm(range(len(batch_files)), desc="Pass 2: Transforming FVs"):
        batch_file = f"batch_{batch_idx:03d}.pkl"
        try:
            original_fvs = fv_dataset_original.load_batch(batch_file)
        except FileNotFoundError:
            continue

        if original_fvs:
            transformed_fvs = {}
            for det_id, fv in original_fvs.items():
                fv_reduced = fv_pca.transform(fv.reshape(1, -1))[0].astype(np.float32)
                transformed_fvs[det_id] = fv_reduced

            fv_dataset_reduced.save_batch(
                transformed_fvs,
                batch_idx,
                metadata={
                    'type': 'weight_fisher_reduced',
                    'gmm_components': K,
                    'pca_components': config.pca.n_components if config.fisher_vector.use_pca else None,
                    'fv_pca_components': config.fisher_vector.fv_pca_components,
                    'original_fv_dimension': fv_dim,
                    'formulation': '2KD+K (weight + mean + variance gradients)',
                }
            )

    # ========== Summary ==========
    print(f"\n{'='*60}")
    print("Weight-Augmented Fisher Vector Processing Complete!")
    print(f"{'='*60}")
    print(f"Total detections processed: {total_processed:,}")
    print(f"Original (normalized) FVs saved to: {fv_dataset_original.fv_root}")
    print(f"Reduced FVs saved to: {fv_dataset_reduced.fv_root}")
    print(f"Raw (unnormalized) FVs saved to: {raw_pkl_path}")
    print(f"Dimension: {fv_dim:,} (2KD+K) -> {config.fisher_vector.fv_pca_components} (PCA reduced)")

    print_memory_summary("End")


if __name__ == "__main__":
    main()
