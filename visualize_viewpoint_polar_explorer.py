"""Interactive viewpoint polar explorer with sprite atlas.

Uses Laplacian eigenvectors (EV1, EV2) in polar coordinates to display
detections by viewpoint angle (atan2) and eigenvector magnitude.

Reuses sprite atlas infrastructure from visualize_tsne_image_explorer.py.

Usage:
    python visualize_viewpoint_polar_explorer.py --config config_zebra_test.yaml
    python visualize_viewpoint_polar_explorer.py --config config_zebra_test.yaml --knn-k 10
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

from src.config.config import MainConfig
from src.data.preprocessed_dataset import PreprocessedDataset
from src.data.coco_loader import COCOLoader
from src.evaluation import load_or_compute_matching, get_identity_mapping

# Reuse sprite atlas functions from the t-SNE explorer
from visualize_tsne_image_explorer import (
    generate_sprite_atlas,
    load_sprite_atlas,
    compute_min_zoom,
    generate_viewer_html,
    THUMBNAIL_SIZE,
)

# Coarse viewpoint mapping and colors
_COARSE_MAP = {
    'right': 'right', 'backright': 'right', 'frontright': 'right',
    'upright': 'right', 'downright': 'right',
    'upbackright': 'right', 'downbackright': 'right', 'upfrontright': 'right',
    'left': 'left', 'backleft': 'left', 'frontleft': 'left',
    'upleft': 'left', 'downleft': 'left',
    'upbackleft': 'left', 'upfrontleft': 'left',
    'front': 'front',
    'back': 'back',
}
VP_COLORS = {
    'right': [30, 120, 180],         # blue
    'backright': [100, 150, 220],    # light blue
    'frontright': [0, 190, 220],     # cyan
    'left': [220, 40, 40],           # red
    'backleft': [250, 130, 130],     # salmon
    'frontleft': [180, 40, 40],      # dark red
    'front': [40, 160, 40],          # green
    'back': [255, 165, 0],           # orange
    'upright': [140, 100, 200],      # purple
    'downright': [180, 180, 40],     # olive
    'upbackright': [0, 180, 180],    # teal
    'downbackright': [128, 128, 128],# gray
    'upfrontright': [255, 180, 120], # peach
    'upleft': [255, 130, 170],       # pink
    'downleft': [160, 100, 200],     # lavender
    'upbackleft': [180, 140, 100],   # tan
    'upfrontleft': [240, 130, 200],  # hot pink
    'unknown': [80, 80, 80],         # dark gray
}


def load_data(config_path: Path):
    """Load Fisher vectors and matching data."""
    print(f"Loading config from: {config_path}")
    config = MainConfig.from_yaml(config_path)
    dataset = PreprocessedDataset(config.output_root)
    coco_loader = COCOLoader(config.coco_json_path, config.dataset_root)

    # Raw weight Fisher vectors
    raw_pkl_path = config.output_root / 'weight_fisher_vectors_raw.pkl'
    print(f'Loading raw weight FVs from {raw_pkl_path}')
    with open(raw_pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    all_det_ids = raw_data['det_ids']
    all_fvs_raw = raw_data['fvs_raw']
    del raw_data

    # GMM info
    from src.codebook.gmm_trainer import load_gmm_model
    gmm, _ = load_gmm_model(config.gmm_model_path)
    K = gmm.n_components
    D = gmm.means_.shape[1]
    block = 2 * D + 1
    del gmm

    # Power + L2 normalize
    signs = np.sign(all_fvs_raw)
    np.abs(all_fvs_raw, out=all_fvs_raw)
    np.sqrt(all_fvs_raw, out=all_fvs_raw)
    all_fvs_raw *= signs
    norms = np.linalg.norm(all_fvs_raw, axis=1, keepdims=True)
    np.maximum(norms, 1e-10, out=norms)
    all_fvs_raw /= norms
    all_fvs_norm = all_fvs_raw

    # Matching
    matched = load_or_compute_matching(
        dataset, coco_loader, config.output_root,
        target_size=config.active_resize_size,
        patch_size=config.active_patch_size,
        category_names=config.matching_categories,
        min_overlap_fraction=0.5,
    )
    identity_map = get_identity_mapping(matched)

    # Viewpoint info
    det_to_viewpoint = {}
    for m in matched:
        det_to_viewpoint[m.detection_id] = m.gt_annotation.viewpoint

    return (all_det_ids, all_fvs_norm, identity_map, det_to_viewpoint,
            dataset, config.output_root, K, block)


def compute_laplacian_polar(fvs: np.ndarray, knn_k: int = 10):
    """Compute kNN graph Laplacian and return polar coordinates from EV1-EV2."""
    N = len(fvs)

    print(f"Building {knn_k}-NN graph on {N} FVs...")
    nn = NearestNeighbors(n_neighbors=knn_k, metric='euclidean', n_jobs=-1)
    nn.fit(fvs)
    _, indices = nn.kneighbors(fvs)

    rows, cols = [], []
    for i in range(N):
        for jj in range(1, knn_k):
            rows.extend([i, indices[i, jj]])
            cols.extend([indices[i, jj], i])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A = (A > 0).astype(np.float64)
    degrees = np.array(A.sum(axis=1)).ravel()

    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
    L_sym = D_inv_sqrt @ (diags(degrees) - A) @ D_inv_sqrt

    print("Computing Laplacian eigenvectors...")
    eigenvalues, eigenvectors = eigsh(L_sym, k=6, which='SM')
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    print(f"  Eigenvalues: {eigenvalues}")

    # Polar coordinates from EV1-EV2
    ev1 = eigenvectors[:, 1]
    ev2 = eigenvectors[:, 2]
    theta = np.arctan2(ev2, ev1)  # viewpoint angle
    r = np.sqrt(ev1**2 + ev2**2)  # magnitude

    # Polar layout: angle = viewpoint, radius = magnitude (no scaling)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    embedding = np.stack([x, y], axis=1)

    return embedding, theta, r, eigenvectors, eigenvalues


def create_viewpoint_server_app(
    det_ids, embedding, identity_map, det_to_viewpoint, fvs_norm,
    atlas_path, atlas_data, port=8081, target_spacing=50.0, overlap_budget=2.0,
):
    """Create Flask server with viewpoint-colored borders."""
    from flask import Flask, send_file, jsonify

    # Uniform scaling to preserve aspect ratio (critical for polar layout)
    emb_center = (embedding.max(axis=0) + embedding.min(axis=0)) / 2
    emb_scale = (embedding.max(axis=0) - embedding.min(axis=0)).max()
    if emb_scale == 0:
        emb_scale = 1.0
    embedding_normalized = (embedding - emb_center) / emb_scale + 0.5

    canvas_size = 2000
    positions = embedding_normalized * canvas_size
    thumbnail_size = atlas_data["thumbnail_size"]

    print("Computing minZoom levels...")
    min_zoom_values = compute_min_zoom(positions, target_spacing_pixels=target_spacing, overlap_budget=overlap_budget)

    points_data = []
    for idx, det_id in enumerate(det_ids):
        identity = identity_map.get(det_id)
        has_identity = identity is not None and identity.lower() != "nan"
        icon_info = atlas_data["icons"][det_id]

        raw_vp = det_to_viewpoint.get(det_id, 'unknown')
        coarse_vp = _COARSE_MAP.get(raw_vp, 'unknown')
        vp_color = VP_COLORS.get(raw_vp, VP_COLORS.get(coarse_vp, VP_COLORS['unknown']))

        points_data.append({
            "idx": idx,
            "det_id": det_id,
            "x": float(positions[idx, 0]),
            "y": float(positions[idx, 1]),
            "identity": identity[:16] + "..." if identity else "unknown",
            "full_identity": identity if identity else "unknown",
            "has_identity": has_identity,
            "icon_col": icon_info["col"],
            "icon_row": icon_info["row"],
            "minZoom": float(min_zoom_values[idx]),
            "viewpoint": raw_vp,
            "coarse_vp": coarse_vp,
            "vp_color": vp_color,
        })

    det_to_image = {det_id: det_id.rsplit('_det_', 1)[0] for det_id in det_ids}

    def find_neighbors(query_idx, k=5, exclude_same_image=True):
        query_fv = fvs_norm[query_idx]
        similarities = fvs_norm @ query_fv
        mask = np.ones(len(det_ids), dtype=bool)
        mask[query_idx] = False
        if exclude_same_image:
            query_image = det_to_image[det_ids[query_idx]]
            for i, det_id in enumerate(det_ids):
                if det_to_image[det_id] == query_image:
                    mask[i] = False
        similarities[~mask] = -np.inf
        top_k = np.argsort(similarities)[-k:][::-1]
        return [{"idx": int(i), "sim": float(similarities[i])} for i in top_k]

    app = Flask(__name__)
    atlas_dir = atlas_path.parent

    @app.route('/')
    def index():
        return send_file(atlas_dir / 'viewpoint_viewer.html')

    @app.route('/atlas.png')
    def serve_atlas():
        return send_file(atlas_path, mimetype='image/png')

    @app.route('/points.json')
    def serve_points():
        return jsonify(points_data)

    @app.route('/neighbors/<int:idx>')
    def get_neighbors(idx):
        if idx < 0 or idx >= len(det_ids):
            return jsonify({"error": "Invalid index"}), 400
        neighbors = find_neighbors(idx, k=5)
        return jsonify(neighbors)

    # Generate HTML with viewpoint border support
    html_content = generate_viewer_html(
        thumbnail_size=thumbnail_size,
        atlas_width=atlas_data["atlas_width"],
        atlas_height=atlas_data["atlas_height"],
        canvas_size=canvas_size,
        base_target_spacing=target_spacing,
        base_overlap_budget=overlap_budget,
    )

    # Inject viewpoint border layer + dots-only toggle
    border_injection = """
        let showDotsOnly = false;

        function createBorderLayer() {
            const filteredPoints = getFilteredPoints();
            return new deck.ScatterplotLayer({
                id: 'vp-borders',
                data: filteredPoints,
                pickable: !showDotsOnly,
                getPosition: d => [d.x, CANVAS_SIZE - d.y],
                getRadius: showDotsOnly ? 8 * sizeScale : 36 * sizeScale,
                radiusUnits: 'pixels',
                getFillColor: d => [...d.vp_color, showDotsOnly ? 220 : 0],
                getLineColor: d => d.vp_color,
                getLineWidth: showDotsOnly ? 0 : 3,
                stroked: !showDotsOnly,
                filled: showDotsOnly,
                lineWidthUnits: 'pixels',
            });
        }
"""
    # Replace updateLayers to include border layer and conditional icon layer
    html_content = html_content.replace(
        "function updateLayers() {\n            const layers = [createIconLayer()];",
        border_injection + "\n        function updateLayers() {\n            const layers = [createBorderLayer()];\n            if (!showDotsOnly) layers.push(createIconLayer());"
    )

    # Add toggle checkbox to controls
    html_content = html_content.replace(
        '<label><input type="checkbox" id="show-identity" checked> With identity</label>',
        '<label><input type="checkbox" id="show-dots-only"> Dots only</label>\n'
        '            <label><input type="checkbox" id="show-identity" checked> With identity</label>'
    )

    # Add event listener for dots toggle
    html_content = html_content.replace(
        "document.getElementById('show-identity').addEventListener",
        "document.getElementById('show-dots-only').addEventListener('change', (e) => {\n"
        "                showDotsOnly = e.target.checked;\n"
        "                updateLayers();\n"
        "                updatePointCount();\n"
        "            });\n\n"
        "            document.getElementById('show-identity').addEventListener"
    )

    # Update tooltip to show viewpoint
    html_content = html_content.replace(
        "getTooltip: ({object}) => object && `${object.identity}\\n${object.det_id}`",
        "getTooltip: ({object}) => object && `${object.viewpoint} (${object.coarse_vp})\\n${object.identity}\\n${object.det_id}`"
    )

    html_path = atlas_dir / "viewpoint_viewer.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Saved HTML viewer to: {html_path}")

    return app


def main():
    parser = argparse.ArgumentParser(description="Viewpoint polar explorer")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--knn-k", type=int, default=10, help="kNN k for graph construction")
    parser.add_argument("--regenerate-atlas", action="store_true", help="Force regenerate sprite atlas")
    parser.add_argument("--thumbnail-size", type=int, default=THUMBNAIL_SIZE)
    parser.add_argument("--target-spacing", type=float, default=50.0)
    parser.add_argument("--overlap-budget", type=float, default=2.0)
    parser.add_argument("--port", type=int, default=8081, help="Port for server")
    args = parser.parse_args()

    # Load data
    (det_ids, fvs_norm, identity_map, det_to_viewpoint,
     dataset, output_root, K, block) = load_data(args.config)

    # Filter to detections with valid viewpoints (same as notebook)
    _COARSE_MAP_FILTER = {
        'right': 'right', 'backright': 'right', 'frontright': 'right',
        'upright': 'right', 'downright': 'right',
        'upbackright': 'right', 'downbackright': 'right', 'upfrontright': 'right',
        'left': 'left', 'backleft': 'left', 'frontleft': 'left',
        'upleft': 'left', 'downleft': 'left',
        'upbackleft': 'left', 'upfrontleft': 'left',
        'front': 'front', 'back': 'back',
    }
    valid_mask = []
    for det_id in det_ids:
        vp = det_to_viewpoint.get(det_id, 'unknown')
        valid_mask.append(vp not in ('ignore', 'unknown') and vp in _COARSE_MAP_FILTER)
    valid_mask = np.array(valid_mask)
    det_ids_list = [d for d, v in zip(det_ids, valid_mask) if v]
    fvs_norm = fvs_norm[valid_mask]
    print(f"Filtered to {len(det_ids_list)} detections with valid viewpoints (from {len(det_ids)})")

    # Sprite atlas — reuse from t-SNE explorer if available
    atlas_dir = output_root / "viewpoint_explorer"
    atlas_dir.mkdir(exist_ok=True)

    det_ids_hash = hashlib.md5("".join(det_ids_list).encode()).hexdigest()[:8]
    atlas_path = atlas_dir / f"atlas_{det_ids_hash}_{args.thumbnail_size}.png"

    # Check t-SNE explorer atlas first (same thumbnails, different embedding)
    tsne_atlas_path = output_root / "tsne_explorer" / f"atlas_{det_ids_hash}_{args.thumbnail_size}.png"

    if atlas_path.exists() and not args.regenerate_atlas:
        print(f"Loading existing atlas: {atlas_path}")
        _, atlas_data = load_sprite_atlas(atlas_path)
    else:
        # Search for ANY existing atlas with matching hash + size
        import shutil
        found_atlas = None
        for explorer_dir in output_root.glob("*_explorer"):
            candidate = explorer_dir / f"atlas_{det_ids_hash}_{args.thumbnail_size}.png"
            if candidate.exists() and candidate != atlas_path:
                found_atlas = candidate
                break

        if found_atlas and not args.regenerate_atlas:
            print(f"Reusing atlas from: {found_atlas}")
            shutil.copy2(found_atlas, atlas_path)
            shutil.copy2(found_atlas.with_suffix('.json'), atlas_path.with_suffix('.json'))
            _, atlas_data = load_sprite_atlas(atlas_path)
        else:
            # Link thumbnail cache from best existing source
            vp_cache = atlas_dir / "thumbnail_cache"
            if not vp_cache.exists():
                candidate_caches = []
                for explorer_dir in output_root.glob("*_explorer"):
                    cache = explorer_dir / "thumbnail_cache"
                    if cache.exists() and cache.is_dir():
                        n_cached = len(list(cache.glob("*.png")))
                        candidate_caches.append((cache, n_cached))
                if candidate_caches:
                    best_cache, best_n = max(candidate_caches, key=lambda x: x[1])
                    print(f"  Linking thumbnail cache: {best_cache} ({best_n} thumbnails)")
                    vp_cache.symlink_to(best_cache)

            print("Generating new sprite atlas...")
            atlas_data = generate_sprite_atlas(
                det_ids_list, dataset, atlas_path, thumbnail_size=args.thumbnail_size
            )

    # Compute Laplacian polar embedding
    embedding, theta, r, eigenvectors, eigenvalues = compute_laplacian_polar(
        fvs_norm, knn_k=args.knn_k
    )

    print(f"\nViewpoint angle range: [{np.degrees(theta.min()):.1f}, {np.degrees(theta.max()):.1f}]°")
    print(f"Magnitude range: [{r.min():.5f}, {r.max():.5f}]")

    # Create and run Flask server with viewpoint borders
    app = create_viewpoint_server_app(
        det_ids_list, embedding, identity_map, det_to_viewpoint, fvs_norm,
        atlas_path, atlas_data, port=args.port,
        target_spacing=args.target_spacing,
        overlap_budget=args.overlap_budget,
    )

    import socket
    hostname = socket.gethostname()
    print(f"\n{'='*50}")
    print(f"Viewpoint Polar Explorer")
    print(f"Server running at: http://{hostname}:{args.port}/")
    print(f"{'='*50}")
    print("Press Ctrl+C to stop")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
