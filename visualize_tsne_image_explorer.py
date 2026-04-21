"""Interactive t-SNE image explorer with sprite atlas for fast pan/zoom.

Pre-generates a thumbnail sprite atlas for GPU-accelerated rendering.
Uses Deck.gl via pydeck for smooth WebGL-based pan/zoom with thousands of images.

Usage:
    python visualize_tsne_image_explorer.py --config config_zebra_test.yaml
    python visualize_tsne_image_explorer.py --config config_zebra_test.yaml --regenerate-atlas
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.config.config import MainConfig, JointConfig
from src.data.preprocessed_dataset import PreprocessedDataset
from src.data.fv_dataset import FisherVectorDataset
from src.data.coco_loader import COCOLoader
from src.evaluation import load_or_compute_matching, get_identity_mapping
from src.visualization import get_crop_bounds


THUMBNAIL_SIZE = 64  # Size of each thumbnail in the atlas
ATLAS_MAX_SIZE = 8192  # Max texture size for WebGL compatibility


def load_data(config_path: Path, filter_singletons: bool = True):
    """Load Fisher vectors, detections, and identity mapping."""
    print(f"Loading config from: {config_path}")

    config_str = config_path.read_text()
    if "semantic_config_path" in config_str:
        config = JointConfig.from_yaml(config_path)
        fv_root = config.output_root / "conditional_fv_reduced"
        semantic_config = MainConfig.from_yaml(config.semantic_config_path)
        dataset = PreprocessedDataset(semantic_config.output_root)
        coco_loader = COCOLoader(semantic_config.coco_json_path, semantic_config.dataset_root)
        output_root = config.output_root
    else:
        config = MainConfig.from_yaml(config_path)
        fv_root = config.output_root / "weight_fisher_vectors_reduced"
        dataset = PreprocessedDataset(config.output_root)
        coco_loader = COCOLoader(config.coco_json_path, config.dataset_root)
        output_root = config.output_root

    print(f"Loading Fisher vectors from: {fv_root}")
    fv_dataset = FisherVectorDataset(fv_root)
    all_det_ids, all_fvs = fv_dataset.get_all_fisher_vectors()
    print(f"Loaded {len(all_det_ids)} Fisher vectors, dim: {all_fvs.shape[1]}")

    print("Matching detections to ground truth...")
    matched = load_or_compute_matching(dataset, coco_loader, config.output_root, target_size=config.active_resize_size, patch_size=config.active_patch_size, category_names=config.matching_categories)
    identity_map = get_identity_mapping(matched)
    print(f"Matched {len(identity_map)} detections to identities")

    if filter_singletons:
        identity_counts = Counter(identity_map.values())
        non_singleton_ids = {ident for ident, count in identity_counts.items() if count >= 2}

        filtered_det_ids = []
        filtered_fvs = []
        for i, det_id in enumerate(all_det_ids):
            identity = identity_map.get(det_id)
            if identity is not None and identity in non_singleton_ids:
                filtered_det_ids.append(det_id)
                filtered_fvs.append(all_fvs[i])

        det_ids = filtered_det_ids
        fvs = np.vstack(filtered_fvs)
        print(f"After filtering singletons: {len(det_ids)} detections")
    else:
        det_ids = list(all_det_ids)
        fvs = all_fvs

    fvs_norm = fvs / np.linalg.norm(fvs, axis=1, keepdims=True)

    return det_ids, fvs, fvs_norm, identity_map, dataset, coco_loader, output_root


def get_detection_crop(dataset: PreprocessedDataset, det_id: str, size: int = THUMBNAIL_SIZE) -> Image.Image:
    """Load and crop a detection image to thumbnail size."""
    detection = dataset.get_detection(det_id)
    if detection is None:
        return Image.new("RGB", (size, size), color=(128, 128, 128))

    img = Image.open(detection.image_path).convert("RGB")
    img_w, img_h = img.size

    x1, y1, x2, y2 = get_crop_bounds(detection.square_crop_bbox, img_w, img_h)
    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize((size, size), Image.Resampling.LANCZOS)
    return crop


def generate_sprite_atlas(
    det_ids: list[str],
    dataset: PreprocessedDataset,
    output_path: Path,
    thumbnail_size: int = THUMBNAIL_SIZE,
    save_interval: int = 500,
) -> dict:
    """Generate a sprite atlas containing all detection thumbnails.

    Saves progress incrementally to a cache directory. If interrupted,
    resume will skip already-generated thumbnails.

    Returns:
        icon_mapping: dict with atlas metadata and icon positions
    """
    n_images = len(det_ids)

    # Calculate grid dimensions
    # Try to make it roughly square, but respect max texture size
    max_per_row = ATLAS_MAX_SIZE // thumbnail_size
    cols = min(max_per_row, math.ceil(math.sqrt(n_images)))
    rows = math.ceil(n_images / cols)

    atlas_width = cols * thumbnail_size
    atlas_height = rows * thumbnail_size

    print(f"Generating sprite atlas: {cols}x{rows} grid = {atlas_width}x{atlas_height} pixels")
    print(f"Thumbnail size: {thumbnail_size}x{thumbnail_size}")

    # Cache directory for incremental saves
    cache_dir = output_path.parent / "thumbnail_cache"
    cache_dir.mkdir(exist_ok=True)

    # Check which thumbnails already exist in cache
    existing = set()
    for f in cache_dir.glob("*.png"):
        existing.add(f.stem)

    to_generate = [det_id for det_id in det_ids if det_id not in existing]
    print(f"Thumbnails cached: {len(existing)}, to generate: {len(to_generate)}")

    # Generate missing thumbnails
    if to_generate:
        for idx, det_id in enumerate(tqdm(to_generate, desc="Generating thumbnails")):
            thumb = get_detection_crop(dataset, det_id, thumbnail_size)
            # Use a safe filename (det_ids may have special chars)
            thumb.save(cache_dir / f"{det_id}.png", "PNG")

            # Periodic status
            if (idx + 1) % save_interval == 0:
                print(f"  Saved {idx + 1}/{len(to_generate)} thumbnails")

    # Assemble atlas from cached thumbnails
    print("Assembling atlas from cached thumbnails...")
    atlas = Image.new("RGB", (atlas_width, atlas_height), color=(50, 50, 50))
    icon_mapping = {}

    for idx, det_id in enumerate(tqdm(det_ids, desc="Assembling atlas")):
        row = idx // cols
        col = idx % cols

        x = col * thumbnail_size
        y = row * thumbnail_size

        thumb_path = cache_dir / f"{det_id}.png"
        if thumb_path.exists():
            thumb = Image.open(thumb_path)
        else:
            # Fallback - shouldn't happen but just in case
            thumb = get_detection_crop(dataset, det_id, thumbnail_size)

        atlas.paste(thumb, (x, y))

        icon_mapping[det_id] = {
            "x": x,
            "y": y,
            "width": thumbnail_size,
            "height": thumbnail_size,
            "col": col,
            "row": row,
        }

    # Save atlas
    atlas.save(output_path, "PNG", optimize=True)
    print(f"Saved atlas to: {output_path}")

    # Save mapping
    mapping_data = {
        "thumbnail_size": thumbnail_size,
        "cols": cols,
        "rows": rows,
        "atlas_width": atlas_width,
        "atlas_height": atlas_height,
        "n_images": n_images,
        "icons": icon_mapping,
    }
    mapping_path = output_path.with_suffix(".json")
    with open(mapping_path, "w") as f:
        json.dump(mapping_data, f)
    print(f"Saved mapping to: {mapping_path}")

    return mapping_data


def load_sprite_atlas(atlas_path: Path) -> tuple[Image.Image, dict]:
    """Load existing sprite atlas and mapping."""
    mapping_path = atlas_path.with_suffix(".json")

    with open(mapping_path) as f:
        mapping_data = json.load(f)

    atlas = Image.open(atlas_path)
    return atlas, mapping_data


def compute_tsne_embedding(
    fvs: np.ndarray,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    early_exaggeration: float = 12.0,
    max_iter: int = 1000,
) -> np.ndarray:
    """Compute t-SNE embedding."""
    print(f"Computing t-SNE (perplexity={perplexity}, lr={learning_rate})...")

    # Normalize for cosine-like behavior
    fvs_norm = fvs / np.linalg.norm(fvs, axis=1, keepdims=True)

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        metric="euclidean",
        init="pca",
        random_state=42,
        max_iter=max_iter,
    )
    embedding = reducer.fit_transform(fvs_norm)
    print(f"t-SNE complete: {embedding.shape}")

    return embedding


def compute_min_zoom(
    positions: np.ndarray,
    target_spacing_pixels: float = 50.0,  # Desired spacing between thumbnails in pixels
    overlap_budget: float = 0.0,  # Zoom levels of overlap to allow (higher = more overlap)
) -> np.ndarray:
    """Compute minimum zoom level for each point to avoid overlap.

    Uses greedy farthest-point ordering, then computes the exact zoom
    at which each point stops overlapping with its nearest visible neighbor.

    Args:
        positions: (N, 2) array in world coordinates (0 to canvas_size)
        target_spacing_pixels: Baseline spacing for calculations
        overlap_budget: How many zoom levels of overlap to allow.
                       0 = no overlap, 2 = allow 4x overlap, 3 = allow 8x overlap

    Returns:
        Array of minZoom values (continuous). Point visible when zoom >= minZoom.
    """
    from scipy.spatial import cKDTree

    n_points = len(positions)
    min_zoom = np.full(n_points, -999.0)  # Will be set for each point
    processed = np.zeros(n_points, dtype=bool)

    # Use farthest point sampling to determine processing order
    # This ensures well-distributed points get processed first (lower minZoom)
    order = []
    min_dists = np.full(n_points, np.inf)

    # Start with point closest to center
    center = positions.mean(axis=0)
    first = np.argmin(np.sum((positions - center) ** 2, axis=1))
    order.append(first)
    processed[first] = True
    min_zoom[first] = -4.0  # Always visible

    # Build order using farthest point sampling
    for _ in range(n_points - 1):
        last_pos = positions[order[-1]]
        dists = np.sum((positions - last_pos) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[processed] = -1
        next_idx = np.argmax(min_dists)
        order.append(next_idx)
        processed[next_idx] = True

    # Now compute minZoom for each point based on distance to nearest earlier point
    processed[:] = False
    tree_points = []
    tree = None

    for i, idx in enumerate(order):
        if i == 0:
            min_zoom[idx] = -4.0  # First point always visible
            tree_points.append(positions[idx])
            tree = cKDTree(tree_points)
        else:
            # Find distance to nearest already-processed point
            dist, _ = tree.query(positions[idx])

            # At what zoom does this point not overlap?
            # Screen distance = world_distance * 2^zoom
            # No overlap when: world_distance * 2^zoom >= target_spacing_pixels
            # zoom >= log2(target_spacing_pixels / world_distance)
            # Subtract overlap_budget to allow more overlap
            if dist > 0:
                min_zoom[idx] = np.log2(target_spacing_pixels / dist) - overlap_budget
            else:
                min_zoom[idx] = 10.0  # Very high, essentially hidden

            # Clamp to reasonable range
            min_zoom[idx] = np.clip(min_zoom[idx], -5.0, 5.0)

            # Add to tree for next iterations
            tree_points.append(positions[idx])
            tree = cKDTree(tree_points)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_points} points...")

    # Print distribution
    bins = [-4, -2, -1, 0, 1, 2, 3, 5]
    hist, _ = np.histogram(min_zoom, bins=bins)
    print(f"minZoom distribution: {list(zip(bins[:-1], hist))}")

    return min_zoom


def create_server_app(
    det_ids: list[str],
    embedding: np.ndarray,
    identity_map: dict,
    fvs_norm: np.ndarray,
    atlas_path: Path,
    atlas_data: dict,
    port: int = 8080,
    target_spacing: float = 50.0,
    overlap_budget: float = 2.0,
):
    """Create Flask server with deck.gl viewer and server-side neighbor computation."""
    from flask import Flask, send_file, jsonify, send_from_directory

    # Normalize embedding to [0, 1] range
    emb_min = embedding.min(axis=0)
    emb_max = embedding.max(axis=0)
    emb_range = emb_max - emb_min
    embedding_normalized = (embedding - emb_min) / emb_range

    canvas_size = 2000
    positions = embedding_normalized * canvas_size

    thumbnail_size = atlas_data["thumbnail_size"]

    # Compute minimum zoom level for each point based on spacing
    print("Computing minZoom levels...")
    min_zoom_values = compute_min_zoom(positions, target_spacing_pixels=target_spacing, overlap_budget=overlap_budget)

    # Prepare points data (small, loads fast)
    points_data = []
    for idx, det_id in enumerate(det_ids):
        identity = identity_map.get(det_id)
        has_identity = identity is not None and identity.lower() != "nan"
        icon_info = atlas_data["icons"][det_id]

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
            "minZoom": float(min_zoom_values[idx]),  # Minimum zoom to show this point
        })

    # Pre-compute image groupings for exclusion
    det_to_image = {det_id: det_id.rsplit('_det_', 1)[0] for det_id in det_ids}

    def find_neighbors(query_idx: int, k: int = 5, exclude_same_image: bool = True):
        """Compute k nearest neighbors server-side."""
        query_fv = fvs_norm[query_idx]
        similarities = fvs_norm @ query_fv

        # Build exclusion mask
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
        return send_file(atlas_dir / 'tsne_viewer.html')

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

    # Generate HTML
    html_content = generate_viewer_html(
        thumbnail_size=thumbnail_size,
        atlas_width=atlas_data["atlas_width"],
        atlas_height=atlas_data["atlas_height"],
        canvas_size=canvas_size,
        base_target_spacing=target_spacing,
        base_overlap_budget=overlap_budget,
    )

    html_path = atlas_dir / "tsne_viewer.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Saved HTML viewer to: {html_path}")

    return app


def generate_viewer_html(
    thumbnail_size: int,
    atlas_width: int,
    atlas_height: int,
    canvas_size: int,
    base_target_spacing: float = 50.0,
    base_overlap_budget: float = 2.0,
) -> str:
    """Generate the HTML viewer that fetches neighbors from server."""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>t-SNE Image Explorer</title>
    <script src="https://unpkg.com/deck.gl@8.9.33/dist.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; overflow: hidden; }}
        #container {{ width: 100vw; height: 100vh; position: relative; background: #1a1a2e; }}
        #info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-width: 350px;
            max-height: 80vh;
            overflow-y: auto;
            display: none;
            z-index: 100;
        }}
        #info-panel.visible {{ display: block; }}
        #info-panel h3 {{ margin: 0 0 10px 0; }}
        #info-panel img {{ max-width: 100%; border-radius: 4px; margin: 5px 0; }}
        #close-panel {{
            position: absolute;
            top: 8px;
            right: 8px;
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            color: #666;
            padding: 0 5px;
            line-height: 1;
        }}
        #close-panel:hover {{ color: #000; }}
        .neighbor {{
            display: inline-block;
            margin: 5px;
            text-align: center;
            vertical-align: top;
        }}
        .neighbor img {{ width: 80px; height: 80px; object-fit: cover; }}
        .neighbor .sim {{ font-size: 11px; color: #666; }}
        .neighbor .match {{ font-weight: bold; }}
        .neighbor .match.yes {{ color: green; }}
        .neighbor .match.no {{ color: red; }}
        #controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 100;
        }}
        #controls label {{ margin-right: 15px; font-size: 13px; display: inline-block; }}
        #controls .slider-row {{ margin-top: 8px; }}
        #controls .slider-row label {{ display: block; margin-bottom: 4px; }}
        #controls input[type="range"] {{ width: 150px; vertical-align: middle; }}
        #stats {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 100;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
            z-index: 200;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="loading">Loading...</div>

    <div id="controls" style="display:none;">
        <div>
            <label><input type="checkbox" id="show-identity" checked> With identity</label>
            <label><input type="checkbox" id="show-no-identity" checked> No identity</label>
        </div>
        <div class="slider-row">
            <label>Size: <input type="range" id="size-scale" min="0.25" max="3" step="0.25" value="1"> <span id="size-value">1x</span></label>
        </div>
        <div class="slider-row">
            <label>Spacing: <input type="range" id="target-spacing" min="10" max="200" step="5" value="{base_target_spacing}"> <span id="spacing-value">{base_target_spacing}px</span></label>
        </div>
        <div class="slider-row">
            <label>Overlap: <input type="range" id="overlap-budget" min="0" max="5" step="0.5" value="{base_overlap_budget}"> <span id="overlap-value">{base_overlap_budget}</span></label>
        </div>
    </div>

    <div id="info-panel">
        <button id="close-panel" title="Close">&times;</button>
        <h3>Selected Detection</h3>
        <div id="query-info"></div>
        <h4>Nearest Neighbors</h4>
        <div id="neighbors"></div>
    </div>

    <div id="stats" style="display:none;">
        <span id="point-count"></span> |
        Scroll to zoom (LOD auto-adjusts), drag to pan, click for details
    </div>

    <script>
        const THUMBNAIL_SIZE = {thumbnail_size};
        const CANVAS_SIZE = {canvas_size};
        const BASE_TARGET_SPACING = {base_target_spacing};
        const BASE_OVERLAP_BUDGET = {base_overlap_budget};

        let POINTS = [];
        let atlasImg = null;
        let deckgl = null;
        let selectedIdx = null;
        let showIdentity = true;
        let showNoIdentity = true;
        let sizeScale = 1.0;
        let currentZoom = -1;
        let targetSpacing = BASE_TARGET_SPACING;
        let overlapBudget = BASE_OVERLAP_BUDGET;

        async function loadData() {{
            try {{
                document.getElementById('loading').textContent = 'Loading points...';
                const pointsResp = await fetch('points.json');
                POINTS = await pointsResp.json();
                console.log('Loaded', POINTS.length, 'points');

                document.getElementById('loading').textContent = 'Loading atlas...';
                atlasImg = new Image();
                atlasImg.crossOrigin = 'anonymous';

                await new Promise((resolve, reject) => {{
                    atlasImg.onload = resolve;
                    atlasImg.onerror = reject;
                    atlasImg.src = 'atlas.png';
                }});
                console.log('Loaded atlas:', atlasImg.width, 'x', atlasImg.height);

                initViewer();
            }} catch (err) {{
                document.getElementById('loading').textContent = 'Error: ' + err.message;
                console.error(err);
            }}
        }}

        function computeEffectiveZoom(zoom, size) {{
            // Adjust zoom based on thumbnail size
            // Larger thumbnails need more spacing, so act like lower zoom
            const sizeAdjustment = Math.log2(size);
            return zoom - sizeAdjustment;
        }}

        function adjustMinZoom(baseMinZoom) {{
            // Adjust minZoom based on current spacing/overlap settings
            // Formula: new = base + log2(newSpacing/baseSpacing) - (newBudget - baseBudget)
            return baseMinZoom + Math.log2(targetSpacing / BASE_TARGET_SPACING) - (overlapBudget - BASE_OVERLAP_BUDGET);
        }}

        function getFilteredPoints() {{
            const effectiveZoom = computeEffectiveZoom(currentZoom, sizeScale);
            return POINTS.filter(p => {{
                const adjMinZoom = adjustMinZoom(p.minZoom);
                if (adjMinZoom > effectiveZoom) return false;
                if (p.has_identity && !showIdentity) return false;
                if (!p.has_identity && !showNoIdentity) return false;
                return true;
            }});
        }}

        function createIconLayer() {{
            const filteredPoints = getFilteredPoints();

            return new deck.IconLayer({{
                id: 'icons',
                data: filteredPoints,
                pickable: true,
                iconAtlas: atlasImg,
                iconMapping: Object.fromEntries(
                    filteredPoints.map(p => [
                        p.det_id,
                        {{
                            x: p.icon_col * THUMBNAIL_SIZE,
                            y: p.icon_row * THUMBNAIL_SIZE,
                            width: THUMBNAIL_SIZE,
                            height: THUMBNAIL_SIZE,
                            mask: false
                        }}
                    ])
                ),
                getIcon: d => d.det_id,
                getPosition: d => [d.x, CANVAS_SIZE - d.y],
                getSize: 64 * sizeScale,  // Size in pixels
                sizeUnits: 'pixels',  // Constant screen size
                onClick: (info) => {{
                    if (info.object) {{
                        showDetails(info.object);
                    }}
                }},
            }});
        }}

        function createHighlightLayer() {{
            if (selectedIdx === null) return null;

            const point = POINTS[selectedIdx];
            return new deck.ScatterplotLayer({{
                id: 'highlight',
                data: [point],
                pickable: false,
                getPosition: d => [d.x, CANVAS_SIZE - d.y],
                getRadius: 36 * sizeScale,
                radiusUnits: 'pixels',
                getFillColor: [0, 0, 0, 0],
                getLineColor: [0, 100, 255],
                getLineWidth: 3,
                stroked: true,
                lineWidthUnits: 'pixels',
            }});
        }}

        function updateLayers() {{
            const layers = [createIconLayer()];
            const highlight = createHighlightLayer();
            if (highlight) layers.push(highlight);
            deckgl.setProps({{ layers }});
        }}

        function updatePointCount() {{
            const visible = getFilteredPoints().length;
            document.getElementById('point-count').textContent = visible + '/' + POINTS.length + ' points';
        }}

        function getThumbnailDataUrl(iconCol, iconRow) {{
            const canvas = document.createElement('canvas');
            canvas.width = THUMBNAIL_SIZE;
            canvas.height = THUMBNAIL_SIZE;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(
                atlasImg,
                iconCol * THUMBNAIL_SIZE, iconRow * THUMBNAIL_SIZE,
                THUMBNAIL_SIZE, THUMBNAIL_SIZE,
                0, 0,
                THUMBNAIL_SIZE, THUMBNAIL_SIZE
            );
            return canvas.toDataURL();
        }}

        async function showDetails(point) {{
            selectedIdx = point.idx;
            updateLayers();

            const panel = document.getElementById('info-panel');
            const queryInfo = document.getElementById('query-info');
            const neighborsDiv = document.getElementById('neighbors');

            // Show query image immediately
            const queryThumb = getThumbnailDataUrl(point.icon_col, point.icon_row);
            queryInfo.innerHTML = `
                <img src="${{queryThumb}}" style="width: 120px; height: 120px;">
                <p><strong>ID:</strong> ${{point.identity}}</p>
                <p style="font-size: 11px; color: #666;">${{point.det_id}}</p>
            `;
            neighborsDiv.innerHTML = '<em>Loading neighbors...</em>';
            panel.classList.add('visible');

            // Fetch neighbors from server
            try {{
                const resp = await fetch(`/neighbors/${{point.idx}}`);
                const neighbors = await resp.json();

                neighborsDiv.innerHTML = neighbors.map((n, i) => {{
                    const np = POINTS[n.idx];
                    const thumb = getThumbnailDataUrl(np.icon_col, np.icon_row);
                    const isMatch = point.full_identity === np.full_identity && point.full_identity !== 'unknown';
                    const matchClass = isMatch ? 'yes' : 'no';
                    const matchSymbol = isMatch ? '✓' : '✗';
                    return `
                        <div class="neighbor">
                            <img src="${{thumb}}">
                            <div class="sim">sim: ${{n.sim.toFixed(3)}}</div>
                            <div class="match ${{matchClass}}">${{matchSymbol}} ${{np.identity}}</div>
                        </div>
                    `;
                }}).join('');
            }} catch (err) {{
                neighborsDiv.innerHTML = '<em style="color:red;">Error loading neighbors</em>';
                console.error(err);
            }}
        }}

        function initViewer() {{
            document.getElementById('loading').style.display = 'none';
            document.getElementById('controls').style.display = 'block';
            document.getElementById('stats').style.display = 'block';

            let lastEffectiveZoom = computeEffectiveZoom(currentZoom, sizeScale);

            deckgl = new deck.DeckGL({{
                container: 'container',
                initialViewState: {{
                    target: [CANVAS_SIZE / 2, CANVAS_SIZE / 2, 0],
                    zoom: -1,
                    minZoom: -4,
                    maxZoom: 5,
                }},
                controller: {{
                    scrollZoom: true,
                    dragPan: true,
                    dragRotate: false,
                    doubleClickZoom: true,
                    touchZoom: true,
                    touchRotate: false,
                }},
                views: new deck.OrthographicView(),
                layers: [createIconLayer()],
                getTooltip: ({{object}}) => object && `${{object.identity}}\\n${{object.det_id}}`,
                onViewStateChange: ({{viewState}}) => {{
                    currentZoom = viewState.zoom;
                    const newEffectiveZoom = computeEffectiveZoom(currentZoom, sizeScale);
                    // Update when zoom changes by 0.5 (significant visible change)
                    if (Math.abs(newEffectiveZoom - lastEffectiveZoom) > 0.3) {{
                        lastEffectiveZoom = newEffectiveZoom;
                        updateLayers();
                        updatePointCount();
                    }}
                }},
            }});

            document.getElementById('show-identity').addEventListener('change', (e) => {{
                showIdentity = e.target.checked;
                updateLayers();
            }});
            document.getElementById('show-no-identity').addEventListener('change', (e) => {{
                showNoIdentity = e.target.checked;
                updateLayers();
            }});
            document.getElementById('size-scale').addEventListener('input', (e) => {{
                sizeScale = parseFloat(e.target.value);
                document.getElementById('size-value').textContent = sizeScale + 'x';
                updateLayers();
                updatePointCount();
            }});
            document.getElementById('target-spacing').addEventListener('input', (e) => {{
                targetSpacing = parseFloat(e.target.value);
                document.getElementById('spacing-value').textContent = targetSpacing + 'px';
                updateLayers();
                updatePointCount();
            }});
            document.getElementById('overlap-budget').addEventListener('input', (e) => {{
                overlapBudget = parseFloat(e.target.value);
                document.getElementById('overlap-value').textContent = overlapBudget;
                updateLayers();
                updatePointCount();
            }});
            document.getElementById('close-panel').addEventListener('click', () => {{
                document.getElementById('info-panel').classList.remove('visible');
                selectedIdx = null;
                updateLayers();
            }});

            // Initial point count
            updatePointCount();
        }}

        loadData();
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="t-SNE image explorer with sprite atlas")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter singleton identities")
    parser.add_argument("--regenerate-atlas", action="store_true", help="Force regenerate sprite atlas")
    parser.add_argument("--perplexity", type=float, default=30, help="t-SNE perplexity")
    parser.add_argument("--thumbnail-size", type=int, default=THUMBNAIL_SIZE, help="Thumbnail size in pixels")
    parser.add_argument("--target-spacing", type=float, default=50.0, help="Target spacing between thumbnails in pixels")
    parser.add_argument("--overlap-budget", type=float, default=2.0, help="Zoom levels of overlap to allow (higher = more overlap)")
    parser.add_argument("--port", type=int, default=8080, help="Port for server")
    args = parser.parse_args()

    # Load data
    det_ids, fvs, fvs_norm, identity_map, dataset, _, output_root = load_data(
        args.config, filter_singletons=not args.no_filter
    )

    # Sprite atlas path
    atlas_dir = output_root / "tsne_explorer"
    atlas_dir.mkdir(exist_ok=True)

    # Create a hash of det_ids to detect if we need to regenerate
    det_ids_hash = hashlib.md5("".join(det_ids).encode()).hexdigest()[:8]
    atlas_path = atlas_dir / f"atlas_{det_ids_hash}_{args.thumbnail_size}.png"

    # Generate or load sprite atlas
    if atlas_path.exists() and not args.regenerate_atlas:
        print(f"Loading existing atlas: {atlas_path}")
        _, atlas_data = load_sprite_atlas(atlas_path)
    else:
        print("Generating new sprite atlas...")
        atlas_data = generate_sprite_atlas(
            det_ids, dataset, atlas_path, thumbnail_size=args.thumbnail_size
        )

    # Compute t-SNE embedding
    embedding = compute_tsne_embedding(fvs, perplexity=args.perplexity)

    # Create and run Flask server
    app = create_server_app(
        det_ids, embedding, identity_map, fvs_norm,
        atlas_path, atlas_data, port=args.port,
        target_spacing=args.target_spacing,
        overlap_budget=args.overlap_budget,
    )

    import socket
    hostname = socket.gethostname()
    print(f"\n{'='*50}")
    print(f"Server running at: http://{hostname}:{args.port}/")
    print(f"{'='*50}")
    print("Press Ctrl+C to stop")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
