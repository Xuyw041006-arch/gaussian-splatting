"""Training-view region descriptor alignment, inspired by (not reproducing) LaGa.

Pure NumPy helpers shared by CLI queries and evaluators. CLIP scores choose
language candidates; ONLY independent affinity matches regions across views
and attaches their retained language descriptors to Gaussians.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from semantic.artifact import pairwise_relevancy


def normalized(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Descriptor/affinity values must be finite")
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def array_fingerprint(values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    descriptor = json.dumps({"shape": values.shape, "dtype": "float32"}, sort_keys=True).encode()
    digest = hashlib.sha256(descriptor)
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_signature(affinity, iteration, clip_model, clip_pretrained, prefixes=(8, 12, 16), ply_sha256=None):
    affinity = np.asarray(affinity, dtype=np.float32)
    prefixes = [int(x) for x in prefixes]
    if (affinity.ndim != 2 or not len(affinity) or len(prefixes) != 3
            or prefixes != sorted(set(prefixes)) or prefixes[0] < 1 or prefixes[-1] != affinity.shape[1]
            or not np.isfinite(affinity).all() or not np.any(np.linalg.norm(affinity, axis=1) > 1e-8)):
        raise ValueError("A nonempty trained independent affinity field is required")
    return {"iteration": int(iteration), "gaussians": len(affinity),
            "affinity_prefix_dimensions": prefixes, "affinity_sha256": array_fingerprint(affinity),
            "clip_model": str(clip_model), "clip_pretrained": str(clip_pretrained),
            "point_cloud_sha256": ply_sha256}


def select_training_views(fit_names, train_names, val_names, test_names, camera_names, max_views=24):
    """Fail closed on missing/ambiguous identities and any held-out leakage."""
    def identities(names):
        stems = [Path(str(name)).stem for name in names]
        if len(stems) != len(set(stems)):
            raise ValueError("Duplicate/ambiguous image stems in descriptor-bank split")
        return set(stems)
    fit, train, val, test, cameras = map(identities, (fit_names, train_names, val_names, test_names, camera_names))
    if (not fit or fit != train or fit & (val | test) or val & test or cameras != train or max_views < 1):
        raise ValueError("Descriptor bank requires exact train-only fit/cameras, disjoint validation/test lists")
    names = sorted(str(name) for name in camera_names)
    count = min(len(names), int(max_views))
    indices = [min(len(names) - 1, int((i + .5) * len(names) / count)) for i in range(count)]
    return [names[i] for i in indices]


def choose_regions(hierarchy, confidences, max_regions):
    """Bounded, deterministic per-level round robin; never use text/GT ranking."""
    hierarchy, confidences = np.asarray(hierarchy), np.asarray(confidences)
    if hierarchy.ndim != 3 or hierarchy.shape[0] != 3 or max_regions < 1:
        raise ValueError("Expected three hierarchy maps and positive region limit")
    groups = []
    for level in hierarchy:
        ids = [int(x) for x in np.unique(level) if x >= 0]
        if ids and max(ids) >= len(confidences):
            raise ValueError("Hierarchy region ID exceeds descriptor count")
        groups.append(sorted(ids, key=lambda index: (-float(confidences[index]), index)))
    chosen = []
    for offset in range(max(map(len, groups), default=0)):
        for group in groups:
            if offset < len(group) and group[offset] not in chosen:
                chosen.append(group[offset])
                if len(chosen) >= max_regions:
                    return chosen
    return chosen


def pool_view_regions(raw, affinity_image, alpha, view_id, prefixes=(8, 12, 16),
                      max_regions=96, max_pixels=512, min_alpha=.05,
                      min_coverage=.35, min_coherence=.10):
    """Pool original overlapping masks, not the lossy exclusive region map."""
    affinity_image, alpha = np.asarray(affinity_image, dtype=np.float32), np.asarray(alpha, dtype=np.float32)
    shape = tuple(int(x) for x in raw["mask_shape"])
    features = np.asarray(raw["features"], dtype=np.float32)
    confidence = np.asarray(raw["confidences"], dtype=np.float32)
    hierarchy = np.asarray(raw["hierarchy_region_maps"])
    packed = np.asarray(raw["packed_region_masks"], dtype=np.uint8)
    if (len(shape) != 2 or min(shape) < 1 or features.ndim != 2 or packed.ndim != 2
            or packed.shape[1] != (int(np.prod(shape)) + 7) // 8
            or affinity_image.shape != (*shape, prefixes[-1]) or alpha.shape != shape
            or hierarchy.shape != (3, *shape) or len(features) != len(packed)
            or confidence.shape != (len(features),) or max_pixels < 1
            or any(not 0 <= value <= 1 for value in (min_alpha, min_coverage, min_coherence))):
        raise ValueError("Incompatible raw masks/descriptors/rendered affinity or sampling options")
    normalized(features)  # Validate without replacing the retained original CLIP descriptors.
    if not np.isfinite(alpha).all() or not np.isfinite(confidence).all() or not np.isfinite(affinity_image).all():
        raise ValueError("Nonfinite teacher confidence, alpha or affinity render")
    records = []
    for region in choose_regions(hierarchy, confidence, max_regions):
        mask = np.unpackbits(packed[region], count=int(np.prod(shape))).reshape(shape).astype(bool)
        mask_pixels = int(mask.sum())
        valid = mask & (alpha >= min_alpha)
        locations = np.flatnonzero(valid.reshape(-1))
        coverage = len(locations) / max(mask_pixels, 1)
        if (not len(locations) or coverage < min_coverage or confidence[region] <= 0
                or np.linalg.norm(features[region]) <= 1e-8):
            continue
        if len(locations) > max_pixels:
            locations = locations[np.linspace(0, len(locations) - 1, max_pixels).astype(int)]
        sampled = affinity_image.reshape(-1, prefixes[-1])[locations]
        for level, dimensions in enumerate(prefixes):
            if not np.any(hierarchy[level] == region):
                continue
            pooled = normalized(sampled[:, :dimensions]).mean(axis=0)
            coherence = float(np.clip(np.linalg.norm(pooled), 0, 1))
            if coherence < min_coherence or coherence <= 1e-8:
                continue
            prototype = np.zeros(prefixes[-1], dtype=np.float32)
            prototype[:dimensions] = pooled / coherence
            records.append({"clip_features": features[region], "affinity_features": prototype,
                            "view_id": int(view_id), "region_id": region, "level": level,
                            "confidence": float(np.clip(confidence[region], 0, 1)),
                            "coverage": coverage, "coherence": coherence,
                            "mask_pixels": mask_pixels, "sampled_pixels": len(locations)})
    return records


def records_to_bank(records, metadata):
    if not records:
        raise ValueError("No supported region descriptors; do not write an empty bank")
    fields = {name: np.stack([record[name] for record in records]) for name in records[0]}
    for name in ("clip_features", "affinity_features"):
        fields[name] = fields[name].astype(np.float16)
    for name in ("view_id", "region_id", "level", "mask_pixels", "sampled_pixels"):
        fields[name] = fields[name].astype(np.int32)
    fields["metadata"] = metadata
    return fields


def save_bank(path, bank):
    """Create one new artifact; never overwrite an existing model/bank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **{key: value for key, value in bank.items() if key != "metadata"},
                            metadata_json=np.asarray(json.dumps(bank["metadata"], sort_keys=True)))


def load_bank(path, signature=None):
    with np.load(path, allow_pickle=False) as payload:
        bank = {key: payload[key].copy() for key in payload.files if key != "metadata_json"}
        bank["metadata"] = json.loads(str(payload["metadata_json"].item()))
    validate_bank(bank, signature)
    return bank


def validate_bank(bank, signature=None):
    metadata = bank["metadata"]
    if metadata.get("schema_version") != 1 or metadata.get("construction") != "training_view_region_alignment":
        raise ValueError("Unsupported descriptor bank protocol")
    if signature is not None and metadata.get("model_signature") != signature:
        raise ValueError("Descriptor bank belongs to different geometry/affinity/iteration/CLIP weights")
    rows = len(bank["clip_features"])
    if not rows or any(len(bank[name]) != rows for name in (
        "affinity_features", "view_id", "region_id", "level", "confidence", "coverage", "coherence")):
        raise ValueError("Descriptor bank array lengths disagree")
    dimensions = metadata.get("model_signature", {}).get("affinity_prefix_dimensions", [])
    if (len(dimensions) != 3 or dimensions != sorted(set(dimensions)) or dimensions[0] < 1
            or np.asarray(bank["clip_features"]).ndim != 2 or np.asarray(bank["affinity_features"]).ndim != 2
            or bank["affinity_features"].shape[1] != dimensions[-1]
            or any(np.asarray(bank[name]).shape != (rows,) for name in (
                "view_id", "region_id", "level", "confidence", "coverage", "coherence"))):
        raise ValueError("Invalid bank feature, hierarchy or reliability dimensions")
    if not np.isin(bank["level"], [0, 1, 2]).all():
        raise ValueError("Invalid descriptor bank hierarchy level")
    if (not np.isfinite(bank["clip_features"]).all() or not np.isfinite(bank["affinity_features"]).all()
            or np.any(np.linalg.norm(bank["clip_features"].astype(np.float32), axis=1) <= 1e-8)
            or np.any(np.linalg.norm(bank["affinity_features"].astype(np.float32), axis=1) <= 1e-8)):
        raise ValueError("Nonfinite or zero descriptor bank features")
    source_views = metadata.get("source_views", [])
    fit = {Path(str(x)).stem for x in metadata.get("fit_image_names", [])}
    forbidden = {Path(str(x)).stem for x in metadata.get("heldout_image_names", []) + metadata.get("test_image_names", [])}
    if (not source_views or len({Path(str(x)).stem for x in source_views}) != len(source_views)
            or not fit or fit & forbidden or any(Path(str(x)).stem not in fit or Path(str(x)).stem in forbidden for x in source_views)
            or np.any(bank["view_id"] < 0) or np.any(bank["view_id"] >= len(source_views))):
        raise ValueError("Descriptor bank source views are not explicitly train-only")


def cross_view_support(bank, level, threshold=.8, chunk_size=256):
    """Count distinct view IDs with same-level affinity, never CLIP similarity."""
    if level not in (0, 1, 2) or not -1 <= threshold <= 1 or chunk_size < 1:
        raise ValueError("Invalid cross-view support parameters")
    indices = np.flatnonzero(bank["level"] == level)
    dimensions = bank["metadata"]["model_signature"]["affinity_prefix_dimensions"][level]
    features = normalized(bank["affinity_features"][indices, :dimensions])
    views = np.asarray(bank["view_id"])[indices]
    support = np.zeros(len(indices), dtype=np.int32)
    for start in range(0, len(indices), chunk_size):
        matches = features[start:start + chunk_size] @ features.T >= threshold
        for view in np.unique(views):
            support[start:start + chunk_size] += np.any(matches[:, views == view], axis=1)
    result = np.zeros(len(bank["level"]), dtype=np.int32)
    result[indices] = support
    return result


def score_descriptor_bank(bank, gaussian_affinity, positive_text, negative_text, level=1,
                          text_threshold=.5, affinity_threshold=.7, cross_view_threshold=.8,
                          min_views=2, max_candidates=64, chunk_size=4096, temperature=10.0):
    """Return scores/support for N Gaussians (or affinity pixels) x Q prompts.

    Evaluation can call this same function with rendered, alpha-normalized
    affinity pixels. It does not read images, masks, labels or ground truth.
    Model/geometry signature must be validated by the caller before querying.
    Final score = maximum confidence*coverage*coherence*text-relevancy among
    affinity-matched candidates, zero unless >= min_views DISTINCT views match.
    """
    validate_bank(bank)
    if (level not in (0, 1, 2) or min_views < 2 or max_candidates < min_views or chunk_size < 1
            or not 0 <= text_threshold <= 1 or not -1 <= affinity_threshold <= 1):
        raise ValueError("Invalid bank query options; cross-view matching needs at least two views")
    signatures = bank["metadata"]["model_signature"]
    dimensions = signatures["affinity_prefix_dimensions"][level]
    values = np.asarray(gaussian_affinity, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != signatures["affinity_prefix_dimensions"][-1]:
        raise ValueError("Query affinity dimension differs from bank")
    positive, negative = normalized(np.atleast_2d(positive_text)), normalized(np.atleast_2d(negative_text))
    descriptors = normalized(bank["clip_features"])
    if positive.shape[1] != descriptors.shape[1] or negative.shape[1] != descriptors.shape[1]:
        raise ValueError("CLIP dimensions differ from bank")
    relevancy = pairwise_relevancy(descriptors @ positive.T, descriptors @ negative.T, temperature)
    support = cross_view_support(bank, level, cross_view_threshold)
    weights = np.asarray(bank["confidence"]) * np.asarray(bank["coverage"]) * np.asarray(bank["coherence"])
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1 + 1e-5):
        raise ValueError("Invalid descriptor reliability weights")
    scores = np.zeros((len(values), len(positive)), dtype=np.float32)
    view_counts = np.zeros_like(scores, dtype=np.uint16)
    diagnostics = []
    for prompt in range(len(positive)):
        selected = np.flatnonzero((bank["level"] == level) & (support >= min_views)
                                  & (relevancy[:, prompt] >= text_threshold) & (weights > 0))
        # Round-robin source views before a strict cap prevents one view's many
        # overlapping SAM masks from consuming every candidate slot.
        ranking = sorted(selected.tolist(), key=lambda i: (-float(relevancy[i, prompt] * weights[i]), i))
        view_rankings = {}
        for index in ranking:
            view_rankings.setdefault(int(bank["view_id"][index]), []).append(index)
        candidates = []
        for offset in range(max(map(len, view_rankings.values()), default=0)):
            for group in view_rankings.values():
                if offset < len(group) and len(candidates) < max_candidates:
                    candidates.append(group[offset])
        diagnostics.append({"eligible_regions": len(selected), "selected_regions": candidates,
                            "selected_view_ids": sorted({int(bank["view_id"][i]) for i in candidates})})
        if len({int(bank["view_id"][i]) for i in candidates}) < min_views:
            continue
        prototypes = normalized(bank["affinity_features"][candidates, :dimensions])
        source_views = bank["view_id"][candidates]
        candidate_scores = relevancy[candidates, prompt] * weights[candidates]
        for start in range(0, len(values), chunk_size):
            part = values[start:start + chunk_size, :dimensions]
            matches = (normalized(part) @ prototypes.T >= affinity_threshold) & (np.linalg.norm(part, axis=1)[:, None] > 1e-8)
            count = sum(np.any(matches[:, source_views == view], axis=1).astype(np.uint16) for view in np.unique(source_views))
            best = np.max(np.where(matches, candidate_scores[None, :], 0), axis=1)
            scores[start:start + len(part), prompt] = np.where(count >= min_views, best, 0)
            view_counts[start:start + len(part), prompt] = count
    return {"scores": scores, "support_views": view_counts, "queries": diagnostics,
            "protocol": {"score_mode": "descriptor_bank", "level": level, "text_threshold": text_threshold,
                         "affinity_threshold": affinity_threshold, "cross_view_threshold": cross_view_threshold,
                         "min_views": min_views, "max_candidates": max_candidates, "temperature": temperature,
                         "aggregation": "reliability_weighted_max_with_distinct_view_support"}}
