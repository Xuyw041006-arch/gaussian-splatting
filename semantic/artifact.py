"""Numerical helpers shared by semantic query and tests."""

import numpy as np

SCORE_MODES = ("legacy_pca_cosine", "clip_cosine", "clip_relevancy")
DEFAULT_NEGATIVE_PROMPTS = ("object", "things", "stuff", "texture")


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


def clip_space_cosines(decoded, text_features, pca_mean, pca_components, chunk_size=65536):
    """Cosine in reconstructed CLIP space, without allocating N x CLIP-dim.

    PCA centering does not preserve cosine. Compute the exact dot/norm of
    ``decoded @ components + mean`` using the small component Gram matrix.
    Information discarded by PCA is not recovered by this operation.
    """
    values = np.asarray(decoded, dtype=np.float32)
    text = np.atleast_2d(np.asarray(text_features, dtype=np.float32)).copy()
    mean = np.asarray(pca_mean, dtype=np.float32)
    components = np.asarray(pca_components, dtype=np.float32)
    if (values.ndim != 2 or components.ndim != 2 or mean.ndim != 1
            or values.shape[1] != components.shape[0]
            or text.shape[1] != mean.shape[0] or components.shape[1] != mean.shape[0]
            or chunk_size < 1):
        raise ValueError("Incompatible PCA/text feature dimensions or chunk size")
    if not all(np.isfinite(value).all() for value in (values, text, mean, components)):
        raise ValueError("Semantic features and PCA metadata must be finite")
    text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-8)
    gram = components @ components.T
    mean_components = components @ mean
    projected_text = components @ text.T
    mean_text = mean @ text.T
    mean_squared = float(mean @ mean)
    result = np.empty((len(values), len(text)), dtype=np.float32)
    for start in range(0, len(values), chunk_size):
        part = values[start:start + chunk_size]
        squared = np.sum((part @ gram) * part, axis=1) + 2 * (part @ mean_components) + mean_squared
        norm = np.sqrt(np.maximum(squared, 0))
        result[start:start + chunk_size] = np.clip(
            (part @ projected_text + mean_text) / np.maximum(norm[:, None], 1e-8), -1, 1
        )
    return result


def pairwise_relevancy(positive_cosines, negative_cosines, temperature=10.0):
    """Minimum positive-vs-negative two-way softmax, as in LERF/LangSplat.

    The most competitive negative determines each positive probability. This
    is not a probability calibrated against human segmentation annotations.
    """
    positive = np.asarray(positive_cosines, dtype=np.float32)
    negative = np.asarray(negative_cosines, dtype=np.float32)
    if (positive.ndim != 2 or negative.ndim != 2 or len(positive) != len(negative)
            or negative.shape[1] < 1 or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Relevancy needs positive/negative matrices and a positive temperature")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Relevancy inputs must be finite")
    margin = temperature * (positive - negative.max(axis=1, keepdims=True))
    return 1.0 / (1.0 + np.exp(-np.clip(margin, -80, 80)))


def text_retrieval_scores(decoded, positive_text, pca_mean, pca_components,
                          mode="legacy_pca_cosine", negative_text=None, temperature=10.0):
    """Shared scoring for per-Gaussian queries and rendered feature evaluation."""
    positive = np.atleast_2d(np.asarray(positive_text, dtype=np.float32))
    if mode == "legacy_pca_cosine":
        queries = project_clip_feature(positive, pca_mean, pca_components)
        return np.stack([cosine_scores(decoded, query) for query in queries], axis=1)
    if mode not in SCORE_MODES:
        raise ValueError(f"Unknown semantic score mode: {mode}")
    if mode == "clip_relevancy":
        if negative_text is None or np.asarray(negative_text).size == 0:
            raise ValueError("clip_relevancy requires generic negative text embeddings")
        negative = np.atleast_2d(np.asarray(negative_text, dtype=np.float32))
        cosines = clip_space_cosines(decoded, np.concatenate([positive, negative]), pca_mean, pca_components)
        return pairwise_relevancy(cosines[:, :len(positive)], cosines[:, len(positive):], temperature)
    return clip_space_cosines(decoded, positive, pca_mean, pca_components)


def affinity_point_scores(features, point_index, prefix_dimensions=(8, 12, 16), level=1):
    """Query an exported independent hierarchy by a picked Gaussian index.

    v5 affinity channels are not language features: no min/max decoding, CLIP,
    PCA or sigmoid is applied. Normalize only the selected coarse/middle/fine
    prefix before taking cosine affinity to the picked point.
    """
    values = np.asarray(features, dtype=np.float32)
    dimensions = tuple(int(value) for value in prefix_dimensions)
    if (values.ndim != 2 or len(dimensions) != 3 or not 0 <= level < 3
            or not 0 <= point_index < len(values)
            or any(not 0 < dim <= values.shape[1] for dim in dimensions)
            or list(dimensions) != sorted(set(dimensions))):
        raise ValueError("Invalid affinity dimensions, granularity or Gaussian index")
    selected = values[:, :dimensions[level]]
    if not np.isfinite(selected).all() or np.linalg.norm(selected[point_index]) < 1e-8:
        raise ValueError("Affinity query needs finite features and a nonzero picked point")
    return cosine_scores(selected, selected[point_index])


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
