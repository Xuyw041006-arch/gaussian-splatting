"""Product-level reconstruction presets and runtime estimates."""

from copy import deepcopy


PRESETS = {
    "quick": {
        "label": "快速",
        "scene_iterations": 7000,
        "feature_dim": 16,
        "feature_width": 256,
        "max_masks": 96,
        "points_per_side": 20,
        "joint_sh_degree": 3,
        "tier_sh_degrees": (1, 2, 3),
        "semantic_start": 700,
        "semantic_ramp_iterations": 1200,
        "joint_semantic_weight": 0.16,
        "semantic_boundary_weight": 0.04,
        "semantic_contrastive_weight": 0.025,
        "semantic_contrastive_samples": 192,
        "semantic_chunks_per_step": 2,
        "semantic_cross_view_weight": 0.04,
        "joint_spatial_samples": 384,
        "densify_until_iter": 5400,
        "densify_grad_threshold": 0.00018,
        "validation_interval": 1000,
    },
    "balanced": {
        "label": "均衡",
        "scene_iterations": 15000,
        "feature_dim": 32,
        "feature_width": 512,
        "max_masks": 192,
        "points_per_side": 32,
        "joint_sh_degree": 5,
        "tier_sh_degrees": (1, 3, 5),
        "semantic_start": 1500,
        "semantic_ramp_iterations": 2500,
        "joint_semantic_weight": 0.22,
        "semantic_boundary_weight": 0.08,
        "semantic_contrastive_weight": 0.05,
        "semantic_contrastive_samples": 320,
        "semantic_chunks_per_step": 3,
        "semantic_cross_view_weight": 0.08,
        "joint_spatial_samples": 768,
        "densify_until_iter": 12000,
        "densify_grad_threshold": 0.00010,
        "validation_interval": 1000,
    },
    "quality": {
        "label": "精度",
        "scene_iterations": 22000,
        "feature_dim": 48,
        "feature_width": 640,
        "max_masks": 256,
        "points_per_side": 40,
        "joint_sh_degree": 5,
        "tier_sh_degrees": (2, 4, 5),
        "semantic_start": 2500,
        "semantic_ramp_iterations": 4000,
        "joint_semantic_weight": 0.26,
        "semantic_boundary_weight": 0.12,
        "semantic_contrastive_weight": 0.07,
        "semantic_contrastive_samples": 448,
        "semantic_chunks_per_step": 4,
        "semantic_cross_view_weight": 0.10,
        "joint_spatial_samples": 1024,
        "densify_until_iter": 17600,
        "densify_grad_threshold": 0.00007,
        "validation_interval": 1000,
    },
}


GPU_SPEED = {"t4": 1.75, "l4": 1.0, "a10g": 0.92, "a100": 0.52}
BASE_TRAIN_MINUTES = {"quick": 38.0, "balanced": 102.0, "quality": 185.0}


def preset(name):
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    return deepcopy(PRESETS[name])


def apply_preset_defaults(args):
    """Fill only unspecified argparse fields, preserving explicit overrides."""
    values = preset(args.preset)
    for key, value in values.items():
        if key == "label" or not hasattr(args, key):
            continue
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def estimate_minutes(name, views, semantics=True, gpu="l4", capture_mode="dense"):
    """Return a conservative (low, high) estimate calibrated on Ramen/L4.

    Training iterations dominate, while SAM/CLIP preprocessing scales with the
    number of images.  Single-image generation is reported separately because
    its diffusion completion cost and uncertainty differ from reconstruction.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    views = max(1, int(views))
    speed = GPU_SPEED.get(str(gpu).lower(), 1.0)
    semantic_per_view = {"quick": 0.11, "balanced": 0.24, "quality": 0.38}[name]
    minutes = BASE_TRAIN_MINUTES[name]
    if semantics:
        minutes += semantic_per_view * views
    else:
        minutes *= 0.72
    if capture_mode == "sparse":
        minutes *= 1.15
    elif capture_mode == "single":
        minutes = 45.0 + minutes * 0.45
    minutes *= speed
    return round(minutes * 0.82), round(minutes * 1.28)

