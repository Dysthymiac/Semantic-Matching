# Animal Re-Identification using Fisher Vectors

A pipeline for animal re-identification using DINOv3 patch features encoded as Fisher Vectors.

## Algorithm Overview

This pipeline implements a Fisher Vector-based approach for animal re-identification:

1. **Detection & Segmentation**: SAM3 detects and segments animals in images using text prompts
2. **Feature Extraction**: DINOv3 extracts dense patch features from detected regions
3. **Dimensionality Reduction**: Incremental PCA reduces feature dimensionality while preserving structure
4. **Codebook Learning**: A Gaussian Mixture Model (GMM) learns a visual vocabulary from patch features
5. **Fisher Vector Encoding**: Each detection is encoded as a Fisher Vector capturing first and second-order statistics of its patches relative to the GMM
6. **Similarity Matching**: Cosine similarity between Fisher Vectors identifies matching individuals

## Project Structure

```
.
├── *.py                    # Main executable scripts
├── config_*.yaml           # Configuration files
├── src/
│   ├── codebook/           # GMM training and loading
│   ├── config/             # Configuration dataclasses
│   ├── data/               # Dataset loaders (COCO, preprocessed, FV)
│   ├── features/           # DINOv3 extractor, Fisher Vector encoder
│   ├── pca/                # Incremental PCA processor
│   ├── pipeline/           # Preprocessing pipeline
│   ├── segmentation/       # SAM3 segmentation
│   ├── utils/              # Memory monitoring, batch storage, sampling
│   └── visualization/      # GMM visualization, plotting primitives
└── notebooks/              # Jupyter notebooks for analysis
```

## Scripts and Execution Order

### 1. Generate Configuration

```bash
python generate_config.py
```

Creates a template configuration file. Edit the generated YAML to set paths and parameters.

### 2. Preprocess Dataset

```bash
python preprocess_dataset.py config.yaml
```

- Runs SAM3 detection on all images
- Extracts DINOv3 patch features for each detection
- Fits incremental PCA on features
- Saves preprocessed dataset to `output_root`

**Output**: `{output_root}/detections/`, `{output_root}/pca/`

### 3. Train GMM Codebook

```bash
python train_gmm.py --config config.yaml
```

- Samples patches from the preprocessed dataset
- Applies PCA transformation
- Trains GMM codebook
- Saves model to `{output_root}/models/`

**Output**: `{output_root}/models/gmm_*.pkl`

### 4. Compute Fisher Vectors

```bash
python compute_fisher_vectors.py --config config.yaml
```

- Loads preprocessed features and GMM
- Computes Fisher Vector for each detection
- Fits PCA on Fisher Vectors for dimensionality reduction
- Saves both original and reduced Fisher Vectors

**Output**: `{output_root}/fisher_vectors_original/`, `{output_root}/fisher_vectors_reduced/`

## Configuration

Key configuration sections:

```yaml
# Paths
dataset_root: "/path/to/images"
output_root: "/path/to/output"
coco_json_path: "/path/to/annotations.json"

# SAM segmentation
sam:
  min_score: 0.9
  species_prompts:
    zebra: ["grevy's zebra", "zebra"]

# DINOv3 features
dino:
  resize_size: 512

# PCA
pca:
  n_components: 256
  whiten: true
  iterative_fitting: true

# GMM
gmm:
  n_components: 64
  covariance_type: "diag"
  n_samples: 1000000

# Fisher Vectors
fisher_vector:
  fv_pca_components: 512
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
uv sync
```
