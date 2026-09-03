"""Geometry-aware helpers for consistent semantic Gaussian features."""

import numpy as np


def build_neighbor_graph(xyz, colors=None, k=8, color_sigma=0.25):
    """Return KNN indices and edge-aware weights for each Gaussian."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2:
        raise ValueError("xyz must have shape [N, 3] with N >= 2")
    if k < 1:
        raise ValueError("k must be positive")
    k = min(int(k), len(xyz) - 1)

    from sklearn.neighbors import NearestNeighbors

    search = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=-1)
    distances, indices = search.fit(xyz).kneighbors(xyz, return_distance=True)
    distances = distances[:, 1:].astype(np.float32)
    indices = indices[:, 1:].astype(np.int64)
    positive = distances[distances > 0]
    spatial_sigma = float(np.median(positive)) * 3.0 if len(positive) else 1.0
    weights = np.exp(-distances / max(spatial_sigma, 1e-8))

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.shape != (len(xyz), 3):
            raise ValueError("colors must have shape [N, 3]")
        color_distance = np.linalg.norm(colors[:, None, :] - colors[indices], axis=-1)
        weights *= np.exp(-color_distance / max(float(color_sigma), 1e-8))
    return indices, np.maximum(weights, 1e-4).astype(np.float32)
