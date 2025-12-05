"""GMM training functions for codebook generation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from ..config.config import GMMConfig
from ..pca.incremental_pca import IncrementalPCAProcessor


def apply_pca_to_patches(
    patches: np.ndarray,
    pca_processor: IncrementalPCAProcessor,
    batch_size: int = 10000
) -> np.ndarray:
    """
    Apply PCA transformation to patches in batches.

    Pure function: Only transforms data.
    """
    print(f"Applying PCA: {patches.shape[1]} → {pca_processor.config.n_components} dimensions")

    transformed_batches = []
    for i in tqdm(range(0, len(patches), batch_size), desc="PCA transform"):
        batch = patches[i:i+batch_size]
        transformed = pca_processor.pca.transform(batch)
        transformed_batches.append(transformed)

    return np.vstack(transformed_batches)


def apply_pca_to_patches_memory_efficient(
    patches: np.ndarray,
    pca_processor: IncrementalPCAProcessor,
    batch_size: int = 10000
) -> np.ndarray:
    """
    Apply PCA transformation to patches with minimal memory usage.

    WARNING: This function modifies the input array to save memory.
    Use only when memory is critical and input data is no longer needed.
    """
    print(f"Applying PCA (memory-efficient): {patches.shape[1]} → {pca_processor.config.n_components} dimensions")

    n_samples = patches.shape[0]
    n_components = pca_processor.config.n_components

    # Pre-allocate output array with float32 to save memory
    transformed = np.empty((n_samples, n_components), dtype=np.float32)

    for i in tqdm(range(0, n_samples, batch_size), desc="PCA transform"):
        end_idx = min(i + batch_size, n_samples)
        batch = patches[i:end_idx]

        # Transform batch directly into output array
        transformed[i:end_idx] = pca_processor.pca.transform(batch).astype(np.float32)

        # Clear the original batch from memory to reduce peak memory usage
        patches[i:end_idx] = 0

    return transformed


def train_gmm(
    patches: np.ndarray,
    config: GMMConfig
) -> GaussianMixture:
    """
    Train GMM on patch features.

    Single responsibility: Only handles GMM training.
    """
    print(f"\nTraining GMM with {config.n_components} components")
    print(f"  Covariance: {config.covariance_type}")
    print(f"  Data shape: {patches.shape}")

    gmm = GaussianMixture(
        n_components=config.n_components,
        covariance_type=config.covariance_type,
        random_state=config.random_seed,
        max_iter=config.max_iter,
        n_init=1,  # Single init for reproducibility
        verbose=2,
        verbose_interval=1
    )

    print("\nFitting GMM...")
    gmm.fit(patches)

    print(f"\nTraining complete:")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")
    print(f"  Lower bound: {gmm.lower_bound_:.2f}")

    return gmm


def calculate_gmm_statistics(
    gmm: GaussianMixture,
    sample_data: np.ndarray = None
) -> dict:
    """
    Calculate GMM model statistics.

    Pure function: Only computes statistics.
    """
    stats = {
        'n_components': gmm.n_components,
        'covariance_type': gmm.covariance_type,
        'converged': gmm.converged_,
        'n_iter': gmm.n_iter_,
        'lower_bound': gmm.lower_bound_
    }

    # Calculate model size
    model_params = (
        gmm.means_.size +
        gmm.covariances_.size +
        gmm.weights_.size
    )
    stats['model_size_mb'] = model_params * 4 / (1024 * 1024)

    # Sample log-likelihood if data provided
    if sample_data is not None:
        sample_size = min(10000, len(sample_data))
        sample_indices = np.random.choice(len(sample_data), size=sample_size, replace=False)
        stats['sample_log_likelihood'] = gmm.score(sample_data[sample_indices])

    return stats


def save_gmm_model(
    gmm: GaussianMixture,
    output_path: Path,
    metadata: dict
) -> None:
    """
    Save GMM model to disk.

    Single responsibility: Only handles model serialization.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'model': gmm,
        'metadata': metadata
    }

    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"GMM saved to: {output_path}")


def load_gmm_model(model_path: Path) -> Tuple[GaussianMixture, dict]:
    """
    Load GMM model from disk.

    Single responsibility: Only handles model deserialization.
    """
    with open(model_path, 'rb') as f:
        save_dict = pickle.load(f)

    return save_dict['model'], save_dict.get('metadata', {})