"""Specialized DINOv3 model loader with direct URL weight loading."""

from __future__ import annotations

import re
import torch
from pathlib import Path
from typing import Dict


def parse_dinov3_weights_file(weights_file: Path) -> Dict[str, str]:
    """Parse the weights file to extract model name -> URL mappings."""
    with open(weights_file, 'r') as f:
        content = f.read()

    weight_urls = {}

    # Regex pattern to match filename followed by URL
    pattern = r'(dinov3_\w+(?:_pretrain_\w+)?)-[a-f0-9]{8}\.pth\s*\n\s*(https://[^\s]+)'

    for match in re.finditer(pattern, content):
        model_name, url = match.groups()
        weight_urls[model_name] = url

    return weight_urls


def load_dinov3_model(model_name: str, device: str, weights_file: Path) -> torch.nn.Module:
    """Load a DINOv3 model with direct URL weight loading."""
    # Parse weights file to get URL
    weight_urls = parse_dinov3_weights_file(weights_file)

    # If exact match not found, search for keys beginning with the model name
    weight_url = None
    if model_name in weight_urls:
        weight_url = weight_urls[model_name]
    else:
        matching_keys = [key for key in weight_urls.keys() if key.startswith(model_name)]
        weight_url = weight_urls[matching_keys[0]]

    # Load model directly from github with URL weights
    # Use the original model_name (not the matched key) as the function name
    model = torch.hub.load(
        'facebookresearch/dinov3',
        model_name,
        source='github',
        weights=weight_url,
        force_reload=False
    )
    model = model.to(torch.device(device))
    model.eval()
    return model