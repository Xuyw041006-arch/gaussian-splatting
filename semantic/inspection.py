"""Projection and picking helpers for click-based object inspection."""

import numpy as np


def project_points(points, camera):
    points = np.asarray(points, dtype=np.float32)
    position = np.asarray(camera["position"], dtype=np.float32)
    camera_to_world_rotation = np.asarray(camera["rotation"], dtype=np.float32)
    camera_points = (points - position) @ camera_to_world_rotation
    depth = camera_points[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
    u = camera["fx"] * camera_points[:, 0] / safe_depth + camera["width"] / 2.0
    v = camera["fy"] * camera_points[:, 1] / safe_depth + camera["height"] / 2.0
    return np.column_stack((u, v)), depth


def pick_point(points, camera, x, y, radius=8.0, candidate_mask=None):
    pixels, depth = project_points(points, camera)
    distance = np.linalg.norm(pixels - np.array([x, y], dtype=np.float32), axis=1)
    valid = (
        (depth > 0) & (distance <= radius)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera["width"])
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera["height"])
    )
    if candidate_mask is not None:
        valid &= np.asarray(candidate_mask, dtype=bool)
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        return None
    depth_scale = max(float(np.median(depth[candidates])), 1e-6)
    cost = depth[candidates] + distance[candidates] * depth_scale / max(radius, 1.0)
    return int(candidates[np.argmin(cost)])
