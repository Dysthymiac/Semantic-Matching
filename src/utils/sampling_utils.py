"""Utilities for sampling patches from preprocessed datasets."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from ..data.preprocessed_dataset import PreprocessedDataset, Detection
from ..utils.batch_storage import load_batch_from_file


def count_patches_per_batch(dataset: PreprocessedDataset) -> List[Tuple]:
    """
    Count valid patches in each batch file.

    Pure function: Only counts, doesn't modify state.
    Returns: List of (batch_path, patch_count, detection_ids) tuples.
    """
    batch_info = []

    for batch_rel_path in tqdm(dataset._index['batch_to_detections'].keys(), desc="Scanning batches"):
        batch_path = dataset.output_root / batch_rel_path
        if not batch_path.exists():
            continue

        detection_ids = dataset._index['batch_to_detections'][batch_rel_path]
        if not detection_ids:
            continue

        # Load batch to count patches
        batch_data = load_batch_from_file(batch_path)
        batch_patches = 0
        valid_detection_ids = []

        for det_id in detection_ids:
            if det_id in batch_data:
                detection = batch_data[det_id]
                if detection.features.numel() > 0 and detection.patch_mask.numel() > 0:
                    n_valid = int(detection.patch_mask.sum().item())
                    if n_valid > 0:
                        batch_patches += n_valid
                        valid_detection_ids.append(det_id)

        if batch_patches > 0:
            batch_info.append((batch_path, batch_patches, valid_detection_ids))

    return batch_info


def map_sample_indices_to_batches(
    batch_info: List[Tuple],
    sample_indices: np.ndarray
) -> dict:
    """
    Map global patch indices to their batch files.

    Pure function: Simple mapping operation.
    Returns: Dict mapping batch_idx -> list of global patch indices.
    """
    # Create patch range mapping
    patch_to_batch = []
    current_idx = 0

    for i, (_, patch_count, _) in enumerate(batch_info):
        patch_to_batch.append((current_idx, current_idx + patch_count, i))
        current_idx += patch_count

    # Map sample indices to batches
    batches_to_sample = {}
    for idx in sample_indices:
        for start_idx, end_idx, batch_idx in patch_to_batch:
            if start_idx <= idx < end_idx:
                if batch_idx not in batches_to_sample:
                    batches_to_sample[batch_idx] = []
                batches_to_sample[batch_idx].append(idx)
                break

    return batches_to_sample


def extract_patches_from_detection(
    detection: Detection,
    indices_to_extract: List[int] = None
) -> np.ndarray:
    """
    Extract valid patches from a detection.

    Pure function: Only extracts specified patches.
    Returns: Array of shape [n_patches, embed_dim].
    """
    # Extract valid patches
    embed_dim, h_patches, w_patches = detection.features.shape
    features_flat = detection.features.view(embed_dim, -1)  # [embed_dim, num_patches]
    patch_mask_flat = detection.patch_mask.flatten().bool()  # [num_patches]
    valid_features = features_flat[:, patch_mask_flat].T.cpu().numpy()  # [valid_patches, embed_dim]

    if indices_to_extract is not None:
        return valid_features[indices_to_extract]
    return valid_features


def sample_patches_from_dataset(
    dataset: PreprocessedDataset,
    n_samples: int,
    random_seed: int = 42,
    pca_processor=None
) -> Tuple[np.ndarray, dict]:
    """
    Randomly sample patches from dataset efficiently.

    Main sampling function that coordinates the process.
    If pca_processor is provided, applies PCA transformation on-the-fly.
    Returns: (sampled_patches, statistics).
    """
    print(f"Sampling {n_samples:,} patches from dataset...")
    if pca_processor:
        print(f"Will apply PCA on-the-fly: 1024 → {pca_processor.config.n_components} dimensions")
    np.random.seed(random_seed)

    # Count patches per batch
    batch_info = count_patches_per_batch(dataset)
    total_patches = sum(count for _, count, _ in batch_info)
    print(f"Found {total_patches:,} total patches across {len(batch_info)} batch files")

    # Determine sampling strategy
    if total_patches <= n_samples:
        print(f"Using all {total_patches:,} patches")
        sample_all = True
        batches_to_sample = None
    else:
        print(f"Randomly sampling {n_samples:,} from {total_patches:,} patches")
        sample_all = False
        sample_indices = np.random.choice(int(total_patches), size=n_samples, replace=False)
        batches_to_sample = map_sample_indices_to_batches(batch_info, sample_indices)

    # Extract patches batch by batch
    sampled_patches = []
    global_patch_counter = 0

    for batch_idx, (batch_path, batch_patch_count, detection_ids) in enumerate(
        tqdm(batch_info, desc="Extracting patches")
    ):
        # Skip if no patches needed from this batch
        if not sample_all and batch_idx not in batches_to_sample:
            global_patch_counter += batch_patch_count
            continue

        # Load batch file once
        batch_data = load_batch_from_file(batch_path)

        # Get indices to sample from this batch
        if not sample_all:
            batch_sample_indices = set(batches_to_sample[batch_idx])

        # Process each detection
        batch_patch_counter = global_patch_counter

        for det_id in detection_ids:
            detection = batch_data[det_id]
            n_valid = int(detection.patch_mask.sum().item())

            if sample_all:
                # Take all patches
                patches = extract_patches_from_detection(detection)
                # Apply PCA if provided
                if pca_processor:
                    patches = pca_processor.pca.transform(patches).astype(np.float32)
                sampled_patches.append(patches)
            else:
                # Determine which patches to take
                patches_to_take = []
                for local_idx in range(int(n_valid)):
                    global_idx = batch_patch_counter + local_idx
                    if global_idx in batch_sample_indices:
                        patches_to_take.append(local_idx)

                if patches_to_take:
                    patches = extract_patches_from_detection(detection, patches_to_take)
                    # Apply PCA if provided
                    if pca_processor:
                        patches = pca_processor.pca.transform(patches).astype(np.float32)
                    sampled_patches.append(patches)

            batch_patch_counter += n_valid

        global_patch_counter = batch_patch_counter

    # Concatenate results
    sampled_array = np.vstack(sampled_patches)

    stats = {
        'total_patches': total_patches,
        'total_batch_files': len(batch_info),
        'sampled_patches': sampled_array.shape[0],
        'feature_dim': sampled_array.shape[1],
        'random_seed': random_seed,
        'pca_applied': pca_processor is not None
    }

    return sampled_array, stats