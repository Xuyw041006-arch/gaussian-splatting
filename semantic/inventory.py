"""Utilities for turning SAM regions and CLIP scores into a scene inventory."""

from collections import defaultdict

import numpy as np


BACKGROUND_HINTS = {
    "background", "wall", "floor", "ceiling", "sky", "ground", "road",
    "grass", "water", "building", "room", "table surface",
}


def normalize_label(value):
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def parse_inventory_config(payload):
    """Read the common user/LLM importance schema.

    Accepted input is either ``{"objects": [...]}`` or a plain list.  Each
    object must have ``label`` and ``tier``/``importance``.
    """
    if isinstance(payload, dict):
        items = payload.get("objects")
        if items is None and isinstance(payload.get("importance"), dict):
            items = payload["importance"].get("objects", [])
        if items is None:
            items = []
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("importance config must contain an objects list")
    result = {"important": [], "normal": [], "background": []}
    for item in items:
        if not isinstance(item, dict) or not item.get("label"):
            raise ValueError("each importance object needs a label")
        tier = str(item.get("tier", item.get("importance", "normal"))).lower()
        if tier not in result:
            raise ValueError(f"invalid importance tier: {tier}")
        label = normalize_label(item["label"])
        aliases = item.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        result[tier].extend([label, *(normalize_label(alias) for alias in aliases)])
    return {key: sorted(set(value)) for key, value in result.items()}


def rank_scene_inventory(
    region_features, candidate_features, candidate_labels, view_ids,
    region_areas=None, threshold=0.22, topk_per_region=2,
):
    """Aggregate per-region CLIP matches into a compact multi-view inventory."""
    region_features = np.asarray(region_features, dtype=np.float32)
    candidate_features = np.asarray(candidate_features, dtype=np.float32)
    view_ids = np.asarray(view_ids)
    if region_features.ndim != 2 or candidate_features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if region_features.shape[1] != candidate_features.shape[1]:
        raise ValueError("region and text feature dimensions must match")
    if len(candidate_labels) != len(candidate_features):
        raise ValueError("candidate label count does not match text features")
    if len(view_ids) != len(region_features):
        raise ValueError("view id count does not match region features")
    if len(region_features) == 0 or len(candidate_features) == 0:
        return []
    areas = (
        np.asarray(region_areas, dtype=np.float32)
        if region_areas is not None else np.ones(len(region_features), dtype=np.float32)
    )
    similarities = region_features @ candidate_features.T
    k = max(1, min(int(topk_per_region), similarities.shape[1]))
    top = np.argpartition(similarities, -k, axis=1)[:, -k:]
    observations = defaultdict(list)
    for region_index, candidates in enumerate(top):
        for candidate_index in candidates:
            score = float(similarities[region_index, candidate_index])
            if score < float(threshold):
                continue
            observations[int(candidate_index)].append(
                (score, str(view_ids[region_index]), float(areas[region_index]))
            )
    inventory = []
    for candidate_index, values in observations.items():
        scores = np.asarray([value[0] for value in values], dtype=np.float32)
        area_values = np.asarray([value[2] for value in values], dtype=np.float32)
        label = normalize_label(candidate_labels[candidate_index])
        inventory.append({
            "label": label,
            "view_count": len({value[1] for value in values}),
            "region_count": len(values),
            "score_max": float(scores.max()),
            "score_mean": float(np.average(scores, weights=np.maximum(area_values, 1e-6))),
            "coverage": float(np.clip(area_values.sum(), 0.0, 1.0)),
            "recommended_tier": "background" if label in BACKGROUND_HINTS else "normal",
        })
    inventory.sort(
        key=lambda item: (item["view_count"], item["score_mean"], item["coverage"]),
        reverse=True,
    )
    return inventory
