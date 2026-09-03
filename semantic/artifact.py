"""Numerical helpers shared by semantic query and tests."""

import numpy as np


def decode_features(encoded, feature_min, feature_max):
    encoded = np.asarray(encoded, dtype=np.float32)
    lower = np.asarray(feature_min, dtype=np.float32)
    upper = np.asarray(feature_max, dtype=np.float32)
    return encoded * (upper - lower) + lower


def project_clip_feature(feature, pca_mean, pca_components):
    feature = np.asarray(feature, dtype=np.float32)
    mean = np.asarray(pca_mean, dtype=np.float32)
    components = np.asarray(pca_components, dtype=np.float32)
    return (feature - mean) @ components.T


def cosine_scores(features, query, epsilon=1e-8):
    features = np.asarray(features, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    feature_norm = np.linalg.norm(features, axis=-1, keepdims=True)
    query_norm = np.linalg.norm(query)
    return (features / np.maximum(feature_norm, epsilon)) @ (query / max(query_norm, epsilon))


def select_indices(scores, threshold, top_k=0):
    scores = np.asarray(scores)
    indices = np.flatnonzero(scores >= threshold)
    if top_k > 0 and len(indices) > top_k:
        order = np.argsort(scores[indices])[-top_k:]
        indices = indices[order]
    return indices[np.argsort(scores[indices])[::-1]]


def apply_scale_gate(encoded, artifact, level=1):
    """Apply a saved SAGA-style scale gate; version-1 artifacts pass through."""
    encoded = np.asarray(encoded, dtype=np.float32)
    if "scale_gate" not in artifact:
        return encoded
    weight = artifact["scale_gate"]["linear.weight"].float().numpy()
    bias = artifact["scale_gate"]["linear.bias"].float().numpy()
    gate = 1.0 / (1.0 + np.exp(
        -(weight[:, 0] * (float(level) / 2.0) + bias)
    ))
    return encoded * gate[None, :]
