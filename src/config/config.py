"""Configuration dataclasses for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Literal

from .base import AutoConfig


@dataclass
class SAMConfig(AutoConfig):
    """SAM3 segmentation configuration."""
    device: str = "cuda"
    min_score: float = 0.9  # Minimum confidence threshold for detections
    species_prompts: Dict[str, list] = field(default_factory=lambda: {
        "zebra": ["grevy's zebra", "grevy"]
    })


@dataclass
class DINOConfig(AutoConfig):
    """DINOv3 feature extraction configuration."""
    model_name: str = "dinov3_vitl16"
    device: str = "cuda"
    resize_size: int = 512
    batch_size: int = 32
    weights_file_path: Path = field(default_factory=lambda: Path.home() / ".dinov3_weights.txt")
    black_threshold: float = 0.25
    extraction_layer: Optional[int] = None  # None = last layer, or specify layer index (0-based)


@dataclass
class DeDoDeConfig(AutoConfig):
    """DeDoDe descriptor extraction configuration.

    Descriptor models:
        - "B-upright": 256-dim, smaller/faster
        - "G-upright": 512-dim, uses DINOv2 ViT-L/14 (recommended for texture)

    Note: resize_size and patch_size match DINOv3 for spatial correspondence.
    Images are internally padded to DINOv2-compatible size (multiple of 14).
    """
    descriptor_weights: str = "G-upright"
    device: str = "cuda"
    resize_size: int = 512  # Match DINOv3 for spatial correspondence
    patch_size: int = 16    # Match DINOv3 grid (32x32 patches)
    batch_size: int = 32
    black_threshold: float = 0.25


@dataclass
class RoMaConfig(AutoConfig):
    """RoMa VGG19-BN fine feature extraction configuration.

    We extract ONLY the VGG19-BN part from RoMa (not DINOv2).
    VGG19 (first 40 layers) outputs features at strides {1, 2, 4, 8}.

    For 512x512 images:
    - stride 1: 512x512, 64 channels
    - stride 2: 256x256, 128 channels
    - stride 4: 128x128, 256 channels
    - stride 8: 64x64, 512 channels (default - downsampled to 32x32)
    """
    device: str = "cuda"
    resize_size: int = 512  # Match DINOv3 for spatial correspondence
    patch_size: int = 16    # Match DINOv3 grid (32x32 patches)
    target_stride: int = 8  # VGG stride to extract (8 = 64x64, downsampled to 32x32)
    batch_size: int = 32
    black_threshold: float = 0.25


@dataclass
class SIFTConfig(AutoConfig):
    """Dense SIFT feature extraction configuration.

    Extracts SIFT descriptors at fixed grid positions matching the 32×32 patch
    grid used by DINOv3/DeDoDe/RoMa for spatial alignment.

    Standard Dense SIFT + Fisher Vector pipeline uses multi-scale extraction.
    Features from all scales are concatenated: 128D * n_scales per keypoint.

    Uses OpenCV SIFT implementation.
    """
    resize_size: int = 512          # Match DINOv3 for spatial correspondence
    patch_size: int = 16            # Match DINOv3 grid (32×32 patches)
    sift_scales: List[float] = field(default_factory=lambda: [16.0, 24.0, 32.0])  # Multi-scale extraction
    contrast_threshold: float = 0.0  # Disable contrast filtering (dense extraction)
    edge_threshold: float = 100.0    # Disable edge filtering
    black_threshold: float = 0.25    # Min ratio of non-black pixels per patch
    batch_size: int = 64             # Can be larger since SIFT runs on CPU


@dataclass
class DISKConfig(AutoConfig):
    """DISK feature extraction configuration.

    Uses kornia's DISK implementation with confidence-weighted descriptors.
    Features are max-pooled by score within semantic grid cells, aligning
    with the 32×32 patch grid used by DINOv3.

    Key properties:
    - Learned detector-descriptor (vs hand-crafted SIFT)
    - Full resolution output, then pooled to 32×32
    - Confidence scores from internal heatmap
    - 128D descriptors, L2 normalized then scaled by confidence
    """
    resize_size: int = 512              # Match DINOv3 for spatial correspondence
    patch_size: int = 16                # Match DINOv3 grid (32×32 patches)
    pretrained_weights: str = "depth"   # Pretrained model: "depth" or other kornia presets
    black_threshold: float = 0.25       # Min ratio of non-black pixels per patch
    batch_size: int = 32


@dataclass
class PCAConfig(AutoConfig):
    """Incremental PCA configuration for feature preprocessing."""
    n_components: int = 256
    batch_size: int = 100
    random_state: int = 42
    whiten: bool = True  # Best practice for Fisher Vectors: whitening helps GMM with diagonal covariance
    svd_solver: str = "auto"
    iterative_fitting: bool = True
    fit_frequency: int = 100


@dataclass
class GMMConfig(AutoConfig):
    """GMM codebook configuration."""
    n_components: int = 256
    covariance_type: str = "diag"
    n_samples: int = 1000000
    random_seed: int = 42
    max_iter: int = 100
    use_pca: bool = True

    @property
    def model_filename(self) -> str:
        """Auto-generate model filename from parameters."""
        pca_suffix = "_pca" if self.use_pca else ""
        return f"gmm_{self.n_components}_{self.covariance_type}{pca_suffix}.pkl"


@dataclass
class FisherVectorConfig(AutoConfig):
    """Fisher Vector encoding configuration."""
    use_pca: bool = True
    fv_pca_components: int = 512
    batch_size: int = 100
    save_interval: int = 100
    pca_fit_samples: int = 1000  # Collect this many FVs before fitting PCA


@dataclass
class BetaVAEConfig(AutoConfig):
    """β-VAE configuration for disentanglement learning.

    Uses Pythae library for training. Higher beta = more disentanglement pressure.
    """
    latent_dim: int = 20
    beta: float = 4.0
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    reconstruction_loss: str = "mse"  # "mse" or "bce"
    device: str = "cuda"
    random_seed: int = 42




@dataclass
class MainConfig(AutoConfig):
    """Main configuration with all sub-configs."""

    # Required paths (with placeholder defaults for template generation)
    dataset_root: Path = Path("/path/to/dataset")
    output_root: Path = Path("/path/to/output")

    # Optional path
    coco_json_path: Optional[Path] = None

    # Processing settings
    processing_batch_size: int = 8
    feature_extractor: Literal["dino", "dedode", "roma", "sift", "disk"] = "dino"

    # Sub-configurations (always present with defaults)
    sam: SAMConfig = field(default_factory=SAMConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    dedode: DeDoDeConfig = field(default_factory=DeDoDeConfig)
    roma: RoMaConfig = field(default_factory=RoMaConfig)
    sift: SIFTConfig = field(default_factory=SIFTConfig)
    disk: DISKConfig = field(default_factory=DISKConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)

    # Optional sub-configurations
    gmm: Optional[GMMConfig] = field(default_factory=GMMConfig)
    fisher_vector: Optional[FisherVectorConfig] = field(default_factory=FisherVectorConfig)
    beta_vae: Optional[BetaVAEConfig] = field(default_factory=BetaVAEConfig)

    @property
    def gmm_model_path(self) -> Optional[Path]:
        """Get full path to GMM model based on config."""
        if self.gmm:
            return self.output_root / "models" / self.gmm.model_filename
        return None

    def validate(self) -> None:
        """Validate configuration consistency."""
        # Check required paths exist (skip validation for placeholder paths)
        if str(self.dataset_root) != "/path/to/dataset" and not self.dataset_root.exists():
            raise ValueError(f"dataset_root does not exist: {self.dataset_root}")

        # Check FV-GMM consistency
        if self.fisher_vector and self.fisher_vector.use_pca:
            if not self.gmm or not self.gmm.use_pca:
                raise ValueError("Fisher Vector expects PCA but GMM not configured with PCA")

    @classmethod
    def generate_template(cls, path: Path = Path("config_template.yaml")) -> None:
        """Generate a template configuration file with all options."""
        config = cls()
        config.to_yaml(path, skip_none=False)
        print(f"Template configuration saved to: {path}")
        print("Please edit the file to set your actual paths and parameters.")


@dataclass
class JointConfig(MainConfig):
    """Configuration for joint semantic + textural features. Extends MainConfig.

    For joint Fisher Vector encoding (compute_joint_fisher_vectors.py):
    - semantic_config_path: Path to semantic pipeline config (output_root loaded from it)
    - textural_config_path: Path to textural pipeline config (output_root loaded from it)
    - power_norm_alpha: Power normalization exponent for joint FVs
    - joint_fv_pca_components: Final PCA dimension for joint FVs
    - joint_fv_pca_fit_samples: Samples to accumulate before fitting joint FV PCA
    """
    semantic_config_path: Optional[Path] = None
    textural_config_path: Optional[Path] = None

    # Joint Fisher Vector encoding parameters
    power_norm_alpha: float = 0.5
    joint_fv_pca_components: int = 2048
    joint_fv_pca_fit_samples: int = 10000

    # Mixture of Linear Regressions parameters (legacy)
    mlr_n_components: int = 64
    mlr_max_iter: int = 100
    mlr_fit_samples: int = 500000
    mlr_tol: float = 1e-4

    # Residual Fisher / GMM parameters
    gmm_n_components: int = 32
    gmm_fit_samples: int = 500000

    # β-VAE disentanglement
    beta_vae: Optional[BetaVAEConfig] = field(default_factory=BetaVAEConfig)