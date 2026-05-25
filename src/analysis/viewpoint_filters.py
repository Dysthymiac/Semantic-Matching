"""Minimal JSON artifacts for reusable viewpoint detection filters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_det_ids(det_ids) -> str:
    """Stable short hash for an ordered detection-id sequence."""
    h = hashlib.sha256()
    for det_id in det_ids:
        h.update(str(det_id).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def _filter_path(path_or_name: str | Path, output_root: str | Path | None = None) -> Path:
    path = Path(path_or_name)
    if path.suffix == ".json" or path.parent != Path("."):
        return path
    if output_root is None:
        raise ValueError("output_root is required when loading a filter by name")
    return Path(output_root) / "viewpoint_filters" / f"{path.name}.json"


def save_viewpoint_filter(
    path: str | Path,
    filter_name: str,
    filtered_det_ids,
    source_det_ids,
    rule_text: str,
    params: dict[str, Any] | None = None,
) -> Path:
    """Save a minimal viewpoint filter artifact as JSON."""
    path = _filter_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    filtered_det_ids = [str(d) for d in filtered_det_ids]
    source_det_ids = [str(d) for d in source_det_ids]
    data = {
        "filter_name": filter_name,
        "filtered_det_ids": filtered_det_ids,
        "source_det_ids_hash": hash_det_ids(source_det_ids),
        "filtered_det_ids_hash": hash_det_ids(filtered_det_ids),
        "n_source": len(source_det_ids),
        "n_kept": len(filtered_det_ids),
        "rule_text": rule_text,
        "params": params or {},
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def load_viewpoint_filter(path_or_name: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    """Load a viewpoint filter artifact by path or by name under output_root."""
    path = _filter_path(path_or_name, output_root)
    with open(path) as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


def apply_viewpoint_filter(
    det_ids,
    *aligned_arrays,
    filtered_det_ids,
    strict: bool = True,
):
    """Apply a saved ordered detection-id filter to det_ids and aligned arrays.

    Returns:
        (filtered_det_ids, *filtered_arrays)
    """
    det_ids = [str(d) for d in det_ids]
    filtered_det_ids = [str(d) for d in filtered_det_ids]
    det_to_idx = {d: i for i, d in enumerate(det_ids)}

    missing = [d for d in filtered_det_ids if d not in det_to_idx]
    if missing and strict:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} filtered detection IDs are missing from current detections; "
            f"first missing: {preview}"
        )

    kept_ids = [d for d in filtered_det_ids if d in det_to_idx]
    indices = [det_to_idx[d] for d in kept_ids]

    filtered_arrays = []
    for arr in aligned_arrays:
        if hasattr(arr, "iloc"):
            filtered_arrays.append(arr.iloc[indices])
        elif hasattr(arr, "shape"):
            filtered_arrays.append(arr[indices])
        else:
            filtered_arrays.append([arr[i] for i in indices])

    return (kept_ids, *filtered_arrays)


def warn_if_source_changed(filter_data: dict[str, Any], current_det_ids) -> None:
    """Print a warning when a filter artifact was built from a different ordered source set."""
    current_hash = hash_det_ids(current_det_ids)
    expected_hash = filter_data.get("source_det_ids_hash")
    if expected_hash and expected_hash != current_hash:
        print(
            "WARNING: viewpoint filter source set hash differs from current detections "
            f"(artifact={expected_hash}, current={current_hash}; "
            f"artifact_n={filter_data.get('n_source')}, current_n={len(current_det_ids)}). "
            "Applying by detection ID."
        )
