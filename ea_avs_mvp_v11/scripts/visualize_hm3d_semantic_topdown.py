#!/usr/bin/env python3
"""Create a top-down HM3D semantic map and furniture position table."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def _labels(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(",")
        if len(fields) >= 4 and fields[0].isdigit():
            result[tuple(bytes.fromhex(fields[1]))] = fields[2].strip('"')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-glb", type=Path, required=True)
    parser.add_argument("--semantic-txt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    color_to_label = _labels(args.semantic_txt)
    scene = trimesh.load(args.semantic_glb, force="scene")
    buckets = defaultdict(list)
    rng = np.random.default_rng(7)
    for geometry_name, mesh in scene.geometry.items():
        uv = getattr(mesh.visual, "uv", None)
        texture = getattr(getattr(mesh.visual, "material", None), "_data", {}).get("baseColorTexture")
        if uv is None or texture is None or len(mesh.faces) == 0:
            continue
        image = np.asarray(texture.convert("RGB"))
        # Semantic textures are categorical RGB colors. Sample triangle centers.
        faces = mesh.faces
        if len(faces) > 20000:
            faces = faces[rng.choice(len(faces), 20000, replace=False)]
        tri_uv = uv[faces].mean(axis=1)
        px = np.clip((tri_uv[:, 0] * (image.shape[1] - 1)).astype(int), 0, image.shape[1] - 1)
        py = np.clip(((1.0 - tri_uv[:, 1]) * (image.shape[0] - 1)).astype(int), 0, image.shape[0] - 1)
        rgb = image[py, px]
        centers = mesh.vertices[faces].mean(axis=1)
        for color in np.unique(rgb, axis=0):
            key = tuple(int(x) for x in color)
            if key not in color_to_label:
                continue
            points = centers[np.all(rgb == color, axis=1)]
            if len(points):
                buckets[color_to_label[key]].append(points)

    furniture_terms = ("bed", "couch", "chair", "table", "cabinet", "wardrobe", "sofa", "desk", "dresser", "shelf", "refrigerator", "oven", "stove", "sink", "toilet", "bath", "fireplace", "nightstand", "countertop")
    positions = []
    for label, chunks in sorted(buckets.items()):
        if not any(term in label.lower() for term in furniture_terms):
            continue
        points = np.concatenate(chunks, axis=0)
        # Separate disconnected instances approximately by source chunks.
        for index, chunk in enumerate(chunks):
            if len(chunk) < 4:
                continue
            low, high = chunk.min(axis=0), chunk.max(axis=0)
            positions.append({"label": label, "instance_index": index, "center_xyz": ((low + high) / 2).round(4).tolist(), "bounds_min_xyz": low.round(4).tolist(), "bounds_max_xyz": high.round(4).tolist(), "num_surface_samples": int(len(chunk))})

    positions_path = args.output_dir / "furniture_positions.json"
    positions_path.write_text(json.dumps({"semantic_glb": str(args.semantic_glb), "objects": positions}, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    palette = {}
    for i, (label, chunks) in enumerate(sorted(buckets.items())):
        if not any(term in label.lower() for term in furniture_terms):
            continue
        color = plt.cm.tab20(i % 20)
        palette[label] = color
        points = np.concatenate(chunks, axis=0)
        ax.scatter(points[:, 0], points[:, 1], s=1.0, alpha=0.55, color=color, label=label)
        center = points[:, :2].mean(axis=0)
        if len(points) > 20:
            ax.text(center[0], center[1], label, fontsize=6, color=color)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("HM3D X (m)")
    ax.set_ylabel("HM3D Y (m)")
    ax.set_title("HM3D 00800 semantic top-down furniture map")
    ax.grid(alpha=0.2)
    if palette:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "topdown_furniture_semantic.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"labels_recovered": len(buckets), "furniture_instances": len(positions), "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
