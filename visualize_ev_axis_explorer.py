"""Interactive Laplacian-EV axis explorer.

Visualizes detections along a chosen Laplacian eigenvector axis. The user
picks an EV index (dropdown) and a reference detection (click or random
button); each detection is then plotted as:

    x = EV[i, k]                                   (chosen EV value)
    y = ||EV[i, ~k] - EV[ref, ~k]|| * y_scale     (distance in OTHER EVs)

Points hugging the X-axis differ from the reference only along EV k —
scanning along the X-axis at low Y reveals what EV k semantically encodes.

Usage:
    python visualize_ev_axis_explorer.py --config config_zebra_test.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from src.config.config import MainConfig
from src.data.annotation_loader import load_annotations
from src.data.preprocessed_dataset import PreprocessedDataset
from src.evaluation import load_or_compute_matching, get_identity_mapping
from src.laplacian import build_knn_graph, normalized_laplacian, smallest_eigenvectors
from src.features.fisher_vector import build_block_mask, normalize_fvs
from src.codebook.gmm_trainer import load_gmm_model

from visualize_tsne_image_explorer import (
    generate_sprite_atlas,
    load_sprite_atlas,
    THUMBNAIL_SIZE,
)

# Coarse viewpoint mapping for filtering (same as polar explorer)
_COARSE_MAP = {
    'right': 'right', 'backright': 'right', 'frontright': 'right',
    'upright': 'right', 'downright': 'right',
    'upbackright': 'right', 'downbackright': 'right', 'upfrontright': 'right',
    'left': 'left', 'backleft': 'left', 'frontleft': 'left',
    'upleft': 'left', 'downleft': 'left',
    'upbackleft': 'left', 'upfrontleft': 'left',
    'front': 'front', 'back': 'back',
}

VP_COLORS = {
    'right': [30, 120, 180], 'backright': [100, 150, 220], 'frontright': [0, 190, 220],
    'left': [220, 40, 40], 'backleft': [250, 130, 130], 'frontleft': [180, 40, 40],
    'front': [40, 160, 40], 'back': [255, 165, 0], 'unknown': [80, 80, 80],
}


def load_data(config_path: Path):
    """Load Fisher vectors, matching, and identity. Mirrors polar explorer."""
    print(f"Loading config from: {config_path}")
    config = MainConfig.from_yaml(config_path)
    dataset = PreprocessedDataset(config.output_root)
    annotation_loader = load_annotations(config)

    raw_pkl_path = config.output_root / 'weight_fisher_vectors_raw.pkl'
    print(f'Loading raw weight FVs from {raw_pkl_path}')
    with open(raw_pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    all_det_ids = raw_data['det_ids']
    all_fvs_raw = raw_data['fvs_raw']
    del raw_data

    gmm, _ = load_gmm_model(config.gmm_model_path)
    K = gmm.n_components
    D = gmm.means_.shape[1]
    del gmm

    fv_mask = build_block_mask(K, D, use_weight=True, use_mean=True, use_var=True)
    all_fvs_norm = normalize_fvs(all_fvs_raw, fv_mask)
    del all_fvs_raw

    matched = load_or_compute_matching(
        dataset, annotation_loader, config.output_root,
        target_size=config.active_resize_size,
        patch_size=config.active_patch_size,
        category_names=config.matching_categories,
        min_overlap_fraction=0.5,
    )
    identity_map = get_identity_mapping(matched)
    det_to_viewpoint = {m.detection_id: m.gt_annotation.viewpoint for m in matched}

    return (all_det_ids, all_fvs_norm, identity_map, det_to_viewpoint,
            dataset, config.output_root)


def filter_to_valid_viewpoints(det_ids, fvs, det_to_viewpoint):
    """Keep only detections with a recognizable viewpoint label (matches polar explorer)."""
    valid_mask = np.array([
        det_to_viewpoint.get(d, 'unknown') in _COARSE_MAP
        for d in det_ids
    ])
    det_ids_filt = [d for d, v in zip(det_ids, valid_mask) if v]
    fvs_filt = fvs[valid_mask]
    print(f"Filtered to {len(det_ids_filt)} detections with valid viewpoints (from {len(det_ids)})")
    return det_ids_filt, fvs_filt


def create_app(
    det_ids,
    eigenvectors,         # [N, n_eig]
    eigenvalues,          # [n_eig]
    identity_map,
    det_to_viewpoint,
    atlas_path,
    atlas_data,
    canvas_size: int = 2000,
    target_spacing: float = 50.0,
    overlap_budget: float = 2.0,
):
    from flask import Flask, send_file, jsonify

    thumbnail_size = atlas_data["thumbnail_size"]

    # Per-detection metadata (sent once to the client)
    points_meta = []
    for idx, det_id in enumerate(det_ids):
        identity = identity_map.get(det_id)
        has_identity = identity is not None and str(identity).lower() != "nan"
        icon_info = atlas_data["icons"][det_id]
        raw_vp = det_to_viewpoint.get(det_id, 'unknown')
        coarse_vp = _COARSE_MAP.get(raw_vp, 'unknown')
        vp_color = VP_COLORS.get(raw_vp, VP_COLORS.get(coarse_vp, VP_COLORS['unknown']))
        points_meta.append({
            "idx": idx,
            "det_id": det_id,
            "identity": (identity[:16] + "...") if (has_identity and len(str(identity)) > 16) else (str(identity) if has_identity else "unknown"),
            "full_identity": str(identity) if has_identity else "unknown",
            "has_identity": has_identity,
            "icon_col": icon_info["col"],
            "icon_row": icon_info["row"],
            "viewpoint": raw_vp,
            "coarse_vp": coarse_vp,
            "vp_color": vp_color,
        })

    # Send eigenvectors as flat float32 binary for compactness; client reshapes.
    n_eig = eigenvectors.shape[1]
    eigvec_blob = eigenvectors.astype(np.float32).tobytes()

    app = Flask(__name__)

    @app.route('/')
    def index():
        return send_file(atlas_path.parent / 'ev_axis_viewer.html')

    @app.route('/atlas.png')
    def serve_atlas():
        return send_file(atlas_path, mimetype='image/png')

    @app.route('/points.json')
    def serve_points():
        return jsonify({
            "points": points_meta,
            "n_eig": n_eig,
            "eigenvalues": [float(v) for v in eigenvalues],
            "atlas_width": atlas_data["atlas_width"],
            "atlas_height": atlas_data["atlas_height"],
            "thumbnail_size": thumbnail_size,
            "canvas_size": canvas_size,
        })

    @app.route('/eigenvectors.bin')
    def serve_eigenvectors():
        from flask import Response
        return Response(eigvec_blob, mimetype='application/octet-stream')

    @app.route('/neighbors/<int:idx>')
    def get_neighbors(idx):
        # Cosine on full eigenvectors as a quick "what's similar overall" check
        if idx < 0 or idx >= len(det_ids):
            return jsonify({"error": "Invalid index"}), 400
        v = eigenvectors[idx]
        v_norm = np.linalg.norm(v) + 1e-10
        sims = (eigenvectors @ v) / (np.linalg.norm(eigenvectors, axis=1) * v_norm + 1e-10)
        sims[idx] = -np.inf
        top = np.argsort(sims)[-5:][::-1]
        return jsonify([{"idx": int(i), "sim": float(sims[i])} for i in top])

    # Write HTML
    html_path = atlas_path.parent / 'ev_axis_viewer.html'
    html_path.write_text(_build_html(
        canvas_size, thumbnail_size, atlas_data,
        base_target_spacing=target_spacing,
        base_overlap_budget=overlap_budget,
    ))
    print(f"Saved HTML viewer to: {html_path}")

    return app


def _build_html(
    canvas_size: int,
    thumbnail_size: int,
    atlas_data: dict,
    base_target_spacing: float = 50.0,
    base_overlap_budget: float = 2.0,
) -> str:
    """The HTML viewer with EV dropdown, reference picker, scale slider, LOD."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Laplacian EV Axis Explorer</title>
    <script src="https://unpkg.com/deck.gl@8.9.33/dist.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; overflow: hidden; }}
        #container {{ width: 100vw; height: 100vh; position: relative; background: #1a1a2e; }}
        #controls {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(255,255,255,0.95); padding: 12px;
            border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 100; max-width: 320px;
        }}
        #controls .row {{ margin-bottom: 8px; }}
        #controls label {{ font-size: 13px; display: inline-block; margin-right: 6px; }}
        #controls select, #controls input[type=range] {{ vertical-align: middle; }}
        #controls input[type=range] {{ width: 160px; }}
        #controls button {{
            font-size: 12px; padding: 4px 8px; cursor: pointer;
            border: 1px solid #999; background: #f0f0f0; border-radius: 4px;
        }}
        #controls button:hover {{ background: #e0e0e0; }}
        #ref-info {{ font-size: 11px; color: #444; margin-top: 6px; max-width: 300px; word-break: break-all; }}
        #info-panel {{
            position: absolute; top: 10px; right: 10px;
            background: rgba(255,255,255,0.95); padding: 15px;
            border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-width: 350px; max-height: 80vh; overflow-y: auto;
            display: none; z-index: 100;
        }}
        #info-panel.visible {{ display: block; }}
        #close-panel {{
            position: absolute; top: 8px; right: 8px;
            background: none; border: none; font-size: 20px;
            cursor: pointer; color: #666; line-height: 1; padding: 0 5px;
        }}
        #stats {{
            position: absolute; bottom: 10px; left: 10px;
            background: rgba(0,0,0,0.7); color: white;
            padding: 8px 12px; border-radius: 4px; font-size: 12px;
            z-index: 100;
        }}
        #loading {{
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            color: white; font-size: 18px; z-index: 200;
        }}
        .neighbor {{ display: inline-block; margin: 5px; text-align: center; vertical-align: top; }}
        .neighbor img {{ width: 80px; height: 80px; object-fit: cover; }}
        .neighbor .sim {{ font-size: 11px; color: #666; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="loading">Loading...</div>

    <div id="controls" style="display:none;">
        <div class="row">
            <label>EV axis (X):
                <select id="ev-select"></select>
            </label>
        </div>
        <div class="row">
            <label>X-scale: <input type="range" id="x-scale" min="0.1" max="20" step="0.1" value="1"> <span id="x-scale-value">1.0x</span></label>
        </div>
        <div class="row">
            <label>Y-scale: <input type="range" id="y-scale" min="0.1" max="20" step="0.1" value="2"> <span id="y-scale-value">2.0x</span></label>
        </div>
        <div class="row">
            <label>Size: <input type="range" id="size-scale" min="0.25" max="3" step="0.25" value="1"> <span id="size-value">1x</span></label>
        </div>
        <div class="row">
            <label>Spacing: <input type="range" id="target-spacing" min="10" max="200" step="5" value="{base_target_spacing}"> <span id="spacing-value">{base_target_spacing}px</span></label>
        </div>
        <div class="row">
            <label>Overlap: <input type="range" id="overlap-budget" min="0" max="5" step="0.5" value="{base_overlap_budget}"> <span id="overlap-value">{base_overlap_budget}</span></label>
        </div>
        <div class="row">
            <label><input type="checkbox" id="mirror-axis" checked> Mirror above/below axis</label>
        </div>
        <div class="row">
            <button id="random-ref">Random reference</button>
            <label><input type="checkbox" id="dots-only"> Dots only</label>
        </div>
        <div id="ref-info">Click a thumbnail to set as reference</div>
    </div>

    <div id="info-panel">
        <button id="close-panel" title="Close">&times;</button>
        <h3>Selected Detection</h3>
        <div id="query-info"></div>
        <div style="margin-top:10px;">
            <button id="set-as-ref">Set as reference</button>
        </div>
    </div>

    <div id="stats" style="display:none;">
        <span id="point-count"></span> | drag to pan, scroll to zoom, click for details
    </div>

    <script>
        const THUMBNAIL_SIZE = {thumbnail_size};
        const CANVAS_SIZE = {canvas_size};
        const ATLAS_WIDTH = {atlas_data["atlas_width"]};
        const ATLAS_HEIGHT = {atlas_data["atlas_height"]};
        const BASE_TARGET_SPACING = {base_target_spacing};
        const BASE_OVERLAP_BUDGET = {base_overlap_budget};

        let META = [];
        let EIGENVECTORS = null;   // Float32Array of length N * n_eig
        let N = 0, n_eig = 0;
        let eigenvalues = [];

        let atlasImg = null;
        let deckgl = null;

        let evIdx = 1;            // initial EV (skip EV0 which is constant)
        let refIdx = 0;
        let xScale = 1.0;
        let yScale = 2.0;
        let dotsOnly = false;
        let mirrorAxis = true;
        let selectedIdx = null;

        // LOD state
        let sizeScale = 1.0;
        let currentZoom = -1;
        let targetSpacing = BASE_TARGET_SPACING;
        let overlapBudget = BASE_OVERLAP_BUDGET;
        let lastEffectiveZoom = -999;
        let signs = null;          // Float32Array length N: ±1 per detection
        let minZoom = null;        // Float32Array length N

        // Cache for current positions
        let positions = null;     // Float32Array length 2N

        function evRow(i) {{
            // Returns the i-th detection's eigenvector slice as a view.
            return EIGENVECTORS.subarray(i * n_eig, (i + 1) * n_eig);
        }}

        function initSigns() {{
            // Deterministic ±1 per detection (Knuth multiplicative hash).
            // Used to mirror points above/below the X-axis purely for
            // visual decongestion — no semantic meaning.
            signs = new Float32Array(N);
            for (let i = 0; i < N; i++) {{
                signs[i] = ((i * 2654435761) & 1) ? 1 : -1;
            }}
        }}

        function recomputePositions() {{
            // x = EV[i, evIdx]
            // y_raw = ||EV[i, ~{{0,evIdx}}] - EV[refIdx, ~{{0,evIdx}}]||
            // (skip EV0 because it's the constant null mode)
            const ref = evRow(refIdx);

            positions = new Float32Array(N * 2);
            let x_min = Infinity, x_max = -Infinity, y_max_raw = 0;
            for (let i = 0; i < N; i++) {{
                const row = evRow(i);
                const x = row[evIdx];
                let s2 = 0;
                for (let j = 0; j < n_eig; j++) {{
                    if (j === evIdx || j === 0) continue;
                    const d = row[j] - ref[j];
                    s2 += d * d;
                }}
                const y_raw = Math.sqrt(s2);
                if (x < x_min) x_min = x;
                if (x > x_max) x_max = x;
                if (y_raw > y_max_raw) y_max_raw = y_raw;
                positions[i*2] = x;
                positions[i*2 + 1] = y_raw;
            }}

            // X centered on reference; xScale controls spread
            const x_ref = ref[evIdx];
            const x_half = Math.max(
                Math.abs(x_max - x_ref),
                Math.abs(x_min - x_ref)
            ) || 1;

            const y_norm = y_max_raw || 1;
            const center_x = CANVAS_SIZE / 2;
            const center_y = CANVAS_SIZE / 2;
            const x_amplitude = (CANVAS_SIZE / 2) * xScale;
            const y_amplitude = (CANVAS_SIZE / 4) * yScale;

            for (let i = 0; i < N; i++) {{
                const x_rel = (positions[i*2] - x_ref) / x_half;
                const x_canvas = center_x + x_rel * x_amplitude;
                const y_offset = (positions[i*2 + 1] / y_norm) * y_amplitude;
                const y_canvas = mirrorAxis
                    ? center_y - signs[i] * y_offset
                    : center_y - y_offset;
                positions[i*2] = x_canvas;
                positions[i*2 + 1] = y_canvas;
            }}
        }}

        function recomputeMinZoom() {{
            // Farthest-point sampling: well-distributed points get assigned
            // first (low minZoom = visible at low zoom). Each subsequent
            // point's minZoom is set from its distance to the nearest
            // already-processed point. In dense clusters only the first
            // sampled point shows at low zoom; the rest reveal as you zoom in.
            //
            // Algorithm runs in a single O(N^2) pass: while doing FPS we
            // maintain `minDistsSq[i]` = squared distance from i to its
            // nearest already-processed point. The next FPS pick is then
            // argmax(minDistsSq), and its minZoom is derived from
            // sqrt(minDistsSq[next]) before we mark it processed.
            minZoom = new Float32Array(N);
            const minDistsSq = new Float32Array(N);
            for (let i = 0; i < N; i++) minDistsSq[i] = Infinity;
            const processed = new Uint8Array(N);

            // First point: closest to centroid (always visible).
            let cx = 0, cy = 0;
            for (let i = 0; i < N; i++) {{ cx += positions[i*2]; cy += positions[i*2+1]; }}
            cx /= N; cy /= N;
            let firstIdx = 0, bestC = Infinity;
            for (let i = 0; i < N; i++) {{
                const dx = positions[i*2] - cx;
                const dy = positions[i*2+1] - cy;
                const d2 = dx*dx + dy*dy;
                if (d2 < bestC) {{ bestC = d2; firstIdx = i; }}
            }}
            minZoom[firstIdx] = -4.0;
            processed[firstIdx] = 1;
            let lastIdx = firstIdx;

            for (let step = 1; step < N; step++) {{
                // Update each unprocessed point's distance to nearest-processed,
                // by considering only the freshly-added `lastIdx`.
                const lx = positions[lastIdx*2];
                const ly = positions[lastIdx*2+1];
                let nextIdx = -1;
                let nextD = -1;
                for (let i = 0; i < N; i++) {{
                    if (processed[i]) continue;
                    const dx = positions[i*2] - lx;
                    const dy = positions[i*2+1] - ly;
                    const d2 = dx*dx + dy*dy;
                    if (d2 < minDistsSq[i]) minDistsSq[i] = d2;
                    if (minDistsSq[i] > nextD) {{ nextD = minDistsSq[i]; nextIdx = i; }}
                }}
                if (nextIdx === -1) break;

                const d = Math.sqrt(minDistsSq[nextIdx]);
                let mz;
                if (d > 0) {{
                    mz = Math.log2(BASE_TARGET_SPACING / d);
                }} else {{
                    mz = 10.0;  // duplicate position
                }}
                if (mz < -5) mz = -5;
                if (mz > 5) mz = 5;
                minZoom[nextIdx] = mz;
                processed[nextIdx] = 1;
                lastIdx = nextIdx;
            }}
        }}

        function computeEffectiveZoom(zoom, size) {{
            return zoom - Math.log2(size);
        }}

        function adjustMinZoom(baseMinZoom) {{
            return baseMinZoom
                + Math.log2(targetSpacing / BASE_TARGET_SPACING)
                - (overlapBudget - BASE_OVERLAP_BUDGET);
        }}

        function isVisible(i) {{
            if (i === refIdx) return true;
            const eff = computeEffectiveZoom(currentZoom, sizeScale);
            return adjustMinZoom(minZoom[i]) <= eff;
        }}

        function getAllPoints() {{
            const out = [];
            for (let i = 0; i < N; i++) {{
                const m = META[i];
                out.push({{
                    idx: i,
                    det_id: m.det_id,
                    x: positions[i*2],
                    y: positions[i*2 + 1],
                    icon_col: m.icon_col,
                    icon_row: m.icon_row,
                    identity: m.identity,
                    full_identity: m.full_identity,
                    has_identity: m.has_identity,
                    viewpoint: m.viewpoint,
                    coarse_vp: m.coarse_vp,
                    vp_color: m.vp_color,
                    is_ref: (i === refIdx),
                }});
            }}
            return out;
        }}

        function getVisiblePoints() {{
            return getAllPoints().filter(p => isVisible(p.idx));
        }}

        function buildLayers() {{
            const points = getVisiblePoints();
            const layers = [];

            // Reference X-axis line (horizontal at refY=center)
            layers.push(new deck.LineLayer({{
                id: 'axis-line',
                data: [{{
                    src: [0, CANVAS_SIZE - (CANVAS_SIZE / 2)],
                    dst: [CANVAS_SIZE, CANVAS_SIZE - (CANVAS_SIZE / 2)],
                }}],
                getSourcePosition: d => d.src,
                getTargetPosition: d => d.dst,
                getColor: [255, 220, 30, 100],
                getWidth: 2,
                widthUnits: 'pixels',
                pickable: false,
            }}));

            // Reference highlight: a big yellow ring at refIdx (always visible)
            const refMeta = META[refIdx];
            layers.push(new deck.ScatterplotLayer({{
                id: 'ref-ring',
                data: [{{ x: positions[refIdx*2], y: positions[refIdx*2+1] }}],
                pickable: false,
                getPosition: d => [d.x, CANVAS_SIZE - d.y],
                getRadius: 60,
                radiusUnits: 'pixels',
                getFillColor: [0, 0, 0, 0],
                getLineColor: [255, 220, 30],
                getLineWidth: 4,
                stroked: true, filled: false,
                lineWidthUnits: 'pixels',
            }}));

            // Border / dot layer (viewpoint color)
            layers.push(new deck.ScatterplotLayer({{
                id: 'borders',
                data: points,
                pickable: !dotsOnly,
                getPosition: d => [d.x, CANVAS_SIZE - d.y],
                getRadius: dotsOnly ? 6 * sizeScale : 30 * sizeScale,
                radiusUnits: 'pixels',
                getFillColor: d => [...d.vp_color, dotsOnly ? 220 : 0],
                getLineColor: d => d.vp_color,
                getLineWidth: dotsOnly ? 0 : 2.5,
                stroked: !dotsOnly,
                filled: dotsOnly,
                lineWidthUnits: 'pixels',
            }}));

            // Icon layer
            if (!dotsOnly) {{
                layers.push(new deck.IconLayer({{
                    id: 'icons',
                    data: points,
                    pickable: true,
                    iconAtlas: atlasImg,
                    iconMapping: Object.fromEntries(
                        points.map(p => [
                            p.det_id,
                            {{
                                x: p.icon_col * THUMBNAIL_SIZE,
                                y: p.icon_row * THUMBNAIL_SIZE,
                                width: THUMBNAIL_SIZE,
                                height: THUMBNAIL_SIZE,
                                mask: false,
                            }}
                        ])
                    ),
                    getIcon: d => d.det_id,
                    getPosition: d => [d.x, CANVAS_SIZE - d.y],
                    getSize: 64 * sizeScale,
                    sizeUnits: 'pixels',
                }}));
            }}

            return layers;
        }}

        function updateLayers() {{
            const layers = buildLayers();
            if (deckgl) deckgl.setProps({{ layers }});
        }}

        function refresh() {{
            // Full recompute: positions changed (EV/ref/yScale/mirror)
            recomputePositions();
            recomputeMinZoom();
            updateLayers();
            updateRefInfo();
            updatePointCount();
        }}

        function refreshLayersOnly() {{
            // Just LOD/size/spacing/overlap changed
            updateLayers();
            updatePointCount();
        }}

        function updateRefInfo() {{
            const m = META[refIdx];
            document.getElementById('ref-info').innerHTML =
                `<b>Reference:</b> idx=${{refIdx}}, ${{m.viewpoint}}<br>` +
                `<small>${{m.det_id}}</small>`;
        }}

        function updatePointCount() {{
            const evVal = eigenvalues[evIdx].toFixed(5);
            const visible = getVisiblePoints().length;
            document.getElementById('point-count').textContent =
                `${{visible}}/${{N}} visible | EV${{evIdx}} (λ=${{evVal}}) | x×${{xScale.toFixed(1)}} y×${{yScale.toFixed(1)}}`;
        }}

        function buildEvDropdown() {{
            const sel = document.getElementById('ev-select');
            sel.innerHTML = '';
            for (let i = 1; i < n_eig; i++) {{
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = `EV${{i}}  (λ=${{eigenvalues[i].toFixed(5)}})`;
                if (i === evIdx) opt.selected = true;
                sel.appendChild(opt);
            }}
        }}

        async function loadData() {{
            try {{
                document.getElementById('loading').textContent = 'Loading metadata...';
                const r = await fetch('points.json');
                const data = await r.json();
                META = data.points;
                N = META.length;
                n_eig = data.n_eig;
                eigenvalues = data.eigenvalues;

                document.getElementById('loading').textContent = 'Loading eigenvectors...';
                const blob = await (await fetch('eigenvectors.bin')).arrayBuffer();
                EIGENVECTORS = new Float32Array(blob);
                if (EIGENVECTORS.length !== N * n_eig) {{
                    throw new Error(`EV size mismatch: got ${{EIGENVECTORS.length}}, expected ${{N*n_eig}}`);
                }}

                document.getElementById('loading').textContent = 'Loading atlas...';
                atlasImg = new Image();
                atlasImg.crossOrigin = 'anonymous';
                await new Promise((resolve, reject) => {{
                    atlasImg.onload = resolve;
                    atlasImg.onerror = reject;
                    atlasImg.src = 'atlas.png';
                }});

                initSigns();
                buildEvDropdown();
                refIdx = Math.floor(Math.random() * N);
                initViewer();
                document.getElementById('loading').style.display = 'none';
                document.getElementById('controls').style.display = 'block';
                document.getElementById('stats').style.display = 'block';
            }} catch (err) {{
                document.getElementById('loading').textContent = 'Error: ' + err.message;
                console.error(err);
            }}
        }}

        function initViewer() {{
            recomputePositions();
            recomputeMinZoom();

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
                layers: buildLayers(),
                getTooltip: ({{object}}) => object && object.det_id
                    ? `${{object.viewpoint}} (${{object.coarse_vp}})\\n${{object.identity}}\\n${{object.det_id}}`
                    : null,
                onClick: (info) => {{
                    if (info && info.object && info.object.det_id) {{
                        selectedIdx = info.object.idx;
                        showDetails(info.object);
                    }}
                }},
                onViewStateChange: ({{viewState}}) => {{
                    currentZoom = viewState.zoom;
                    const eff = computeEffectiveZoom(currentZoom, sizeScale);
                    if (Math.abs(eff - lastEffectiveZoom) > 0.3) {{
                        lastEffectiveZoom = eff;
                        refreshLayersOnly();
                    }}
                }},
            }});

            updateRefInfo();
            updatePointCount();

            // EV dropdown / Y-scale / mirror — change positions, recompute minZoom
            document.getElementById('ev-select').addEventListener('change', (e) => {{
                evIdx = parseInt(e.target.value, 10);
                refresh();
            }});
            document.getElementById('x-scale').addEventListener('input', (e) => {{
                xScale = parseFloat(e.target.value);
                document.getElementById('x-scale-value').textContent = xScale.toFixed(1) + 'x';
                refresh();
            }});
            document.getElementById('y-scale').addEventListener('input', (e) => {{
                yScale = parseFloat(e.target.value);
                document.getElementById('y-scale-value').textContent = yScale.toFixed(1) + 'x';
                refresh();
            }});
            document.getElementById('mirror-axis').addEventListener('change', (e) => {{
                mirrorAxis = e.target.checked;
                refresh();
            }});

            // Reference change
            document.getElementById('random-ref').addEventListener('click', () => {{
                refIdx = Math.floor(Math.random() * N);
                refresh();
            }});
            document.getElementById('set-as-ref').addEventListener('click', () => {{
                if (selectedIdx !== null) {{
                    refIdx = selectedIdx;
                    document.getElementById('info-panel').classList.remove('visible');
                    refresh();
                }}
            }});

            // LOD-only controls (positions don't change, just visibility)
            document.getElementById('size-scale').addEventListener('input', (e) => {{
                sizeScale = parseFloat(e.target.value);
                document.getElementById('size-value').textContent = sizeScale + 'x';
                refreshLayersOnly();
            }});
            document.getElementById('target-spacing').addEventListener('input', (e) => {{
                targetSpacing = parseFloat(e.target.value);
                document.getElementById('spacing-value').textContent = targetSpacing + 'px';
                refreshLayersOnly();
            }});
            document.getElementById('overlap-budget').addEventListener('input', (e) => {{
                overlapBudget = parseFloat(e.target.value);
                document.getElementById('overlap-value').textContent = overlapBudget;
                refreshLayersOnly();
            }});
            document.getElementById('dots-only').addEventListener('change', (e) => {{
                dotsOnly = e.target.checked;
                refreshLayersOnly();
            }});

            document.getElementById('close-panel').addEventListener('click', () => {{
                document.getElementById('info-panel').classList.remove('visible');
            }});
        }}

        async function showDetails(obj) {{
            const panel = document.getElementById('info-panel');
            panel.classList.add('visible');
            const m = META[obj.idx];
            const evVal = EIGENVECTORS[obj.idx * n_eig + evIdx];
            document.getElementById('query-info').innerHTML =
                `<div><b>idx:</b> ${{obj.idx}}</div>` +
                `<div><b>det_id:</b> <small>${{m.det_id}}</small></div>` +
                `<div><b>viewpoint:</b> ${{m.viewpoint}} (${{m.coarse_vp}})</div>` +
                `<div><b>identity:</b> ${{m.full_identity}}</div>` +
                `<div><b>EV${{evIdx}}:</b> ${{evVal.toFixed(5)}}</div>`;
        }}

        loadData();
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Laplacian EV-axis explorer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--n-eig", type=int, default=20, help="Number of EVs to compute")
    parser.add_argument("--regenerate-atlas", action="store_true")
    parser.add_argument("--thumbnail-size", type=int, default=THUMBNAIL_SIZE)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--target-spacing", type=float, default=50.0)
    parser.add_argument("--overlap-budget", type=float, default=2.0)
    args = parser.parse_args()

    det_ids, fvs, identity_map, det_to_viewpoint, dataset, output_root = load_data(args.config)
    det_ids, fvs = filter_to_valid_viewpoints(det_ids, fvs, det_to_viewpoint)

    # Sprite atlas (reuse other explorers' atlas if hash matches)
    atlas_dir = output_root / "ev_axis_explorer"
    atlas_dir.mkdir(exist_ok=True)
    det_ids_hash = hashlib.md5("".join(det_ids).encode()).hexdigest()[:8]
    atlas_path = atlas_dir / f"atlas_{det_ids_hash}_{args.thumbnail_size}.png"

    if atlas_path.exists() and not args.regenerate_atlas:
        print(f"Loading existing atlas: {atlas_path}")
        _, atlas_data = load_sprite_atlas(atlas_path)
    else:
        import shutil
        found_atlas = None
        for explorer_dir in output_root.glob("*_explorer"):
            cand = explorer_dir / f"atlas_{det_ids_hash}_{args.thumbnail_size}.png"
            if cand.exists() and cand != atlas_path:
                found_atlas = cand
                break
        if found_atlas:
            print(f"Reusing atlas from: {found_atlas}")
            shutil.copy2(found_atlas, atlas_path)
            shutil.copy2(found_atlas.with_suffix('.json'), atlas_path.with_suffix('.json'))
            _, atlas_data = load_sprite_atlas(atlas_path)
        else:
            cache = atlas_dir / "thumbnail_cache"
            if not cache.exists():
                # Try to symlink from any existing thumbnail cache
                best = None
                best_n = -1
                for explorer_dir in output_root.glob("*_explorer"):
                    c = explorer_dir / "thumbnail_cache"
                    if c.exists() and c.is_dir():
                        n = len(list(c.glob("*.png")))
                        if n > best_n:
                            best, best_n = c, n
                if best is not None:
                    print(f"Linking thumbnail cache: {best} ({best_n} thumbnails)")
                    cache.symlink_to(best)
            print("Generating new sprite atlas...")
            atlas_data = generate_sprite_atlas(
                det_ids, dataset, atlas_path, thumbnail_size=args.thumbnail_size,
            )

    # Compute Laplacian eigenvectors using src.laplacian
    print(f"\nBuilding {args.knn_k}-NN graph on {len(fvs)} FVs...")
    A, degrees = build_knn_graph(fvs, k=args.knn_k)
    L_sym = normalized_laplacian(A, degrees)
    print(f"Computing {args.n_eig} smallest Laplacian eigenvectors...")
    eigenvalues, eigenvectors = smallest_eigenvectors(L_sym, n_eig=args.n_eig)

    app = create_app(
        det_ids, eigenvectors, eigenvalues, identity_map, det_to_viewpoint,
        atlas_path, atlas_data,
        target_spacing=args.target_spacing,
        overlap_budget=args.overlap_budget,
    )

    import socket
    hostname = socket.gethostname()
    print(f"\n{'='*50}")
    print(f"Laplacian EV-Axis Explorer")
    print(f"Server running at: http://{hostname}:{args.port}/")
    print(f"{'='*50}")
    print("Press Ctrl+C to stop")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
