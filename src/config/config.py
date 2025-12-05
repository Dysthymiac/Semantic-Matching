"""Configuration dataclasses for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

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
class MainConfig(AutoConfig):
    """Main configuration with all sub-configs."""

    # Required paths (with placeholder defaults for template generation)
    dataset_root: Path = Path("/path/to/dataset")
    output_root: Path = Path("/path/to/output")

    # Optional path
    coco_json_path: Optional[Path] = None

    # Processing settings
    processing_batch_size: int = 8

    # Sub-configurations (always present with defaults)
    sam: SAMConfig = field(default_factory=SAMConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)

    # Optional sub-configurations
    gmm: Optional[GMMConfig] = field(default_factory=GMMConfig)
    fisher_vector: Optional[FisherVectorConfig] = field(default_factory=FisherVectorConfig)

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
    """Configuration for joint semantic + textural features. Extends MainConfig."""
    semantic_config_path: Optional[Path] = None
    textural_config_path: Optional[Path] = None