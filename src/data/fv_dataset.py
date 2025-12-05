"""Fisher Vector dataset management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import numpy as np


class FisherVectorDataset:
    """Dataset for managing Fisher Vector encodings."""

    def __init__(self, fv_root: Path) -> None:
        """Initialize Fisher Vector dataset."""
        self.fv_root = Path(fv_root)
        self.index_path = self.fv_root / "index.json"

        # Create directories
        self.fv_root.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self._index = self._load_or_create_index()

    def _load_or_create_index(self) -> dict:
        """Load existing index or create new one."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {
            'detections': {},  # detection_id -> batch_file
            'batches': {},     # batch_file -> list of detection_ids
            'metadata': {
                'total_detections': 0,
                'total_batches': 0,
                'fv_dim': None,
                'gmm_components': None,
                'pca_components': None
            }
        }

    def save_index(self) -> None:
        """Save index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self._index, f, indent=2)

    def save_batch(
        self,
        batch_data: Dict[str, np.ndarray],
        batch_id: int,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Save a batch of Fisher Vectors.

        Args:
            batch_data: Dict mapping detection_id -> fisher_vector
            batch_id: Batch identifier
            metadata: Optional metadata to store
        """
        if not batch_data:
            return

        batch_file = f"batch_{batch_id:03d}.pkl"
        batch_path = self.fv_root / batch_file

        # Update index
        detection_ids = list(batch_data.keys())
        self._index['batches'][batch_file] = detection_ids
        for det_id in detection_ids:
            self._index['detections'][det_id] = batch_file

        # Update metadata
        if batch_data:
            first_fv = next(iter(batch_data.values()))
            self._index['metadata']['fv_dim'] = first_fv.shape[0]

        self._index['metadata']['total_detections'] = len(self._index['detections'])
        self._index['metadata']['total_batches'] = len(self._index['batches'])

        if metadata:
            self._index['metadata'].update(metadata)

        # Save batch
        save_data = {
            'fisher_vectors': batch_data,
            'metadata': metadata or {}
        }

        with open(batch_path, 'wb') as f:
            pickle.dump(save_data, f)

        # Save updated index
        self.save_index()

    def load_batch(self, batch_file: str) -> Dict[str, np.ndarray]:
        """Load a batch of Fisher Vectors."""
        batch_path = self.fv_root / batch_file

        with open(batch_path, 'rb') as f:
            data = pickle.load(f)

        return data['fisher_vectors']

    def get_fisher_vector(self, detection_id: str) -> Optional[np.ndarray]:
        """Get Fisher Vector for a specific detection."""
        if detection_id not in self._index['detections']:
            return None

        batch_file = self._index['detections'][detection_id]
        batch_data = self.load_batch(batch_file)

        return batch_data.get(detection_id)

    def get_all_fisher_vectors(self) -> Tuple[List[str], np.ndarray]:
        """
        Load all Fisher Vectors.

        Returns:
            Tuple of (detection_ids, fisher_vectors_array)
        """
        all_ids = []
        all_fvs = []

        for batch_file in sorted(self._index['batches'].keys()):
            batch_data = self.load_batch(batch_file)
            for det_id in self._index['batches'][batch_file]:
                if det_id in batch_data:
                    all_ids.append(det_id)
                    all_fvs.append(batch_data[det_id])

        if all_fvs:
            return all_ids, np.vstack(all_fvs)
        return [], np.array([])

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return self._index['metadata'].copy()