#!/usr/bin/env python
"""Generate a template configuration file with all available options."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.config import MainConfig


def main():
    """Generate template configuration file."""
    parser = argparse.ArgumentParser(
        description="Generate a template configuration file for the preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default config template
  python generate_config.py

  # Generate config with custom name
  python generate_config.py --output my_config.yaml

  # Force overwrite existing config
  python generate_config.py --output config.yaml --force

After generation, edit the file to set:
  - dataset_root: Path to your image dataset
  - coco_json_path: Path to COCO annotations (optional)
  - output_root: Where to save processed data
  - Adjust other parameters as needed
        """
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("config_template.yaml"),
        help="Output path for the config file (default: config_template.yaml)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing config file"
    )

    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Generate minimal config without optional sections (GMM, Fisher Vector)"
    )

    args = parser.parse_args()

    # Check if file exists
    if args.output.exists() and not args.force:
        print(f"Error: Config file already exists: {args.output}")
        print("Use --force to overwrite or choose a different name")
        return 1

    # Generate config
    print(f"Generating {'minimal' if args.minimal else 'full'} config template...")

    if args.minimal:
        # Create config without optional sections
        config = MainConfig(
            dataset_root=Path("/path/to/dataset"),
            output_root=Path("/path/to/output"),
            coco_json_path=Path("/path/to/annotations.json"),
            gmm=None,
            fisher_vector=None
        )
    else:
        # Create full config with all sections
        config = MainConfig()

    # Save to file
    config.to_yaml(args.output, skip_none=False)

    print(f"✓ Config template saved to: {args.output}")
    print("\nNext steps:")
    print("1. Edit the config file to set your actual paths:")
    print(f"   vim {args.output}")
    print("\n2. Run preprocessing:")
    print(f"   python preprocess_dataset.py {args.output}")

    if not args.minimal:
        print("\n3. Train PCA (if not already done):")
        print(f"   python train_pca.py {args.output}")
        print("\n4. Train GMM for Fisher Vectors:")
        print(f"   python train_gmm.py --config {args.output}")
        print("\n5. Compute Fisher Vectors:")
        print(f"   python compute_fisher_vectors.py --config {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())