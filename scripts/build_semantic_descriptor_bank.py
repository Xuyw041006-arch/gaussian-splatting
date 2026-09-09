"""Build a bounded post-training multiview region bank from v5 affinity.

Not LaGa's adaptive object clustering: overlapping training-view SAM regions
retain their original CLIP descriptors and independent affinity alignment.
No validation/test images or ground-truth labels are loaded by this builder.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from semantic.descriptor_bank import (
    file_fingerprint, model_signature, pool_view_regions, records_to_bank,
    save_bank, select_training_views, validate_bank,
)


def read_names(path):
    values = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"Required split is empty: {path}")
    return values


def training_camera_infos(source, image_subdir, fit_names):
    """Read COLMAP poses, filter identities BEFORE any RGB image is opened."""
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text
    from scene.dataset_readers import readColmapCameras
    source = Path(source)
    if (source / "sparse/0/images.bin").is_file():
        extrinsics = read_extrinsics_binary(source / "sparse/0/images.bin")
        intrinsics = read_intrinsics_binary(source / "sparse/0/cameras.bin")
    else:
        extrinsics = read_extrinsics_text(source / "sparse/0/images.txt")
        intrinsics = read_intrinsics_text(source / "sparse/0/cameras.txt")
    names = {Path(str(name)).stem for name in fit_names}
    filtered = {key: value for key, value in extrinsics.items() if Path(value.name).stem in names}
    return readColmapCameras(filtered, intrinsics, None, str(source / image_subdir), "", [])


def render_affinity_image(camera, gaussians, pipeline, affinity, height, width):
    import torch
    from gaussian_renderer import render

    old_size = camera.image_height, camera.image_width
    camera.image_height, camera.image_width = height, width
    background = torch.zeros(3, device="cuda", dtype=torch.float32)
    try:
        with torch.no_grad():
            alpha = render(camera, gaussians, pipeline, background,
                           override_color=torch.ones((len(affinity), 3), device="cuda"),
                           detach_geometry=True, clamp_output=False)["render"][0]
            image = np.empty((height, width, affinity.shape[1]), dtype=np.float32)
            for start in range(0, affinity.shape[1], 3):
                size = min(3, affinity.shape[1] - start)
                colors = torch.zeros((len(affinity), 3), device="cuda")
                colors[:, :size] = torch.from_numpy(affinity[:, start:start + size]).cuda()
                field = render(camera, gaussians, pipeline, background, override_color=colors,
                               detach_geometry=True, clamp_output=False)["render"][:size]
                field = field / alpha.clamp_min(1e-4)
                image[:, :, start:start + size] = field.permute(1, 2, 0).cpu().numpy()
            return image, alpha.cpu().numpy()
    finally:
        camera.image_height, camera.image_width = old_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", required=True, help="New .npz path; existing paths are never overwritten")
    parser.add_argument("--max_views", type=int, default=24)
    parser.add_argument("--max_regions_per_view", type=int, default=96)
    parser.add_argument("--max_pixels_per_region", type=int, default=512)
    parser.add_argument("--max_records", type=int, default=8192)
    parser.add_argument("--min_alpha", type=float, default=.05)
    parser.add_argument("--min_coverage", type=float, default=.35)
    parser.add_argument("--min_coherence", type=float, default=.10)
    args = parser.parse_args()
    if min(args.iteration, args.max_views, args.max_regions_per_view, args.max_pixels_per_region, args.max_records) < 1:
        parser.error("Iteration and all sample limits must be positive")
    if any(not 0 <= value <= 1 for value in (args.min_alpha, args.min_coverage, args.min_coherence)):
        parser.error("Coverage, alpha and coherence limits must be within [0,1]")
    # A strict preflight bound prevents later views/levels being silently dropped.
    if args.max_views * args.max_regions_per_view * 3 > args.max_records:
        parser.error("max_records must cover max_views * max_regions_per_view * 3; lower explicit sampling limits")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output already exists; preserve it and choose a new path")
    import torch
    if not torch.cuda.is_available():
        parser.error("Building the descriptor bank requires CUDA rendering")
    from interactive_renderer import extract_dataset_and_pipeline, read_model_config
    from scene import GaussianModel
    from utils.camera_utils import loadCam

    started = time.monotonic()
    model = Path(args.model).resolve()
    artifact_path = model / "semantic" / f"iteration_{args.iteration}" / "semantic_features.pt"
    ply = model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if (artifact.get("semantic_protocol") != "v5" or "affinity_features" not in artifact
            or "affinity_prefix_dimensions" not in artifact or artifact.get("scene_iteration") != args.iteration):
        parser.error("A matching v5 export with an independent affinity field is required")
    if (float(artifact.get("affinity_weight", 0)) <= 0
            or args.iteration <= int(artifact.get("semantic_start", args.iteration))):
        parser.error("Affinity training was disabled or export predates its start; cannot build a meaningful bank")
    affinity = artifact["affinity_features"].float().numpy()
    signature = model_signature(affinity, args.iteration, artifact["clip_model"], artifact["clip_pretrained"],
                                artifact["affinity_prefix_dimensions"], file_fingerprint(ply))
    config = read_model_config(model)
    config.eval = True
    source = Path(config.source_path).resolve()
    config.validation_file = str(source / "sparse/0/val.txt")
    # Load only metadata and split identities before any camera/image loading.
    with np.load(source / "semantic_meta.npz", allow_pickle=False) as data:
        meta = {key: data[key].copy() for key in data.files}
    fit = meta["fit_image_names"].tolist()
    heldout = meta["heldout_image_names"].tolist()
    train, val, test = [read_names(source / "sparse/0" / f"{name}.txt") for name in ("train", "val", "test")]
    if ({Path(str(x)).stem for x in heldout} != {Path(x).stem for x in val}
            or str(meta["clip_model"].item()) != artifact["clip_model"]
            or str(meta["clip_pretrained"].item()) != artifact["clip_pretrained"]
            or int(meta.get("teacher_preprocessing_version", 0)) != 2
            or str(meta.get("hierarchy_method", "")) != "containment"
            or str(meta.get("prototype_mode", "")) != "off"
            or str(meta.get("importance_policy", "")) != "competitive_v1"):
        parser.error("Teacher validation split, CLIP model or raw cache version disagrees with v5")
    for key in ("pca_components", "pca_mean", "feature_min", "feature_max"):
        trained_array = artifact[key].detach().float().cpu().numpy()
        if not np.array_equal(trained_array, np.asarray(meta[key], dtype=np.float32)):
            parser.error(f"Teacher {key} differs from the trained artifact; do not mix regenerated teachers")
    select_training_views(fit, train, val, test, train, args.max_views)
    reference_path = model / "semantic_space_reference.json"
    reference_verified = False
    if reference_path.is_file():
        from scripts.run_ramen_recovery import semantic_space_fingerprint
        reference = json.loads(reference_path.read_text())
        if reference.get("fingerprint") != semantic_space_fingerprint(source)["fingerprint"]:
            parser.error("Current teacher metadata/splits differ from the trained model's frozen reference")
        reference_verified = True
    dataset, pipeline = extract_dataset_and_pipeline(config)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(str(ply))
    infos = training_camera_infos(source, dataset.images, fit)
    selected = select_training_views(fit, train, val, test, [info.image_name for info in infos], args.max_views)
    lookup = {info.image_name: info for info in infos}
    if len(gaussians.get_xyz) != len(affinity):
        parser.error("Geometry/affinity Gaussian count mismatch")
    rows, sources = [], []
    for view_id, name in enumerate(selected):
        raw_path = source / "semantic_raw" / f"{Path(name).stem}.npz"
        raw_stamp = (raw_path.stat().st_size, raw_path.stat().st_mtime_ns)
        raw_sha256 = file_fingerprint(raw_path)
        with np.load(raw_path, allow_pickle=False) as data:
            raw = {key: data[key].copy() for key in data.files}
        if Path(str(raw["image_name"].item())).stem != Path(name).stem:
            raise ValueError(f"Raw teacher identity mismatch: {raw_path}")
        if raw["features"].shape[1] != artifact["pca_components"].shape[1]:
            raise ValueError("Raw region descriptors do not have the trained CLIP dimension")
        # Raw descriptors are original CLIP, but their masks/hierarchy must be
        # those used to train this affinity field, not a newly regenerated SAM.
        map_path = source / "semantic_maps" / f"{Path(name).stem}.npz"
        with np.load(map_path, allow_pickle=False) as trained:
            if not np.array_equal(trained["hierarchy_region_ids"], raw["hierarchy_region_maps"]):
                raise ValueError(f"Raw hierarchy differs from the training teacher: {raw_path}")
        height, width = (int(value) for value in raw["mask_shape"])
        # Only this already-validated TRAIN image is opened; release it each view.
        dataset.resolution = width
        camera = loadCam(dataset, view_id, lookup[name], 1.0, False, False)
        image, alpha = render_affinity_image(camera, gaussians, pipeline, affinity, height, width)
        records = pool_view_regions(raw, image, alpha, view_id, signature["affinity_prefix_dimensions"],
                                    args.max_regions_per_view, args.max_pixels_per_region,
                                    args.min_alpha, args.min_coverage, args.min_coherence)
        rows.extend(records)
        if len(rows) > args.max_records:
            raise RuntimeError("Descriptor record limit exceeded; no output was written")
        if raw_stamp != (raw_path.stat().st_size, raw_path.stat().st_mtime_ns):
            raise RuntimeError(f"Raw descriptor cache changed while building: {raw_path}")
        sources.append({"view": name, "raw_path": str(raw_path), "raw_sha256": raw_sha256,
                        "training_map_sha256": file_fingerprint(map_path),
                        "records": len(records), "width": width, "height": height})
        print(json.dumps({"view": name, "records": len(records), "total_records": len(rows)}), flush=True)
        del image, alpha, raw, camera
    metadata = {
        "schema_version": 1, "construction": "training_view_region_alignment",
        "not_official_laga_adaptive_object_clustering": True,
        "model_signature": signature, "model": str(model), "source_scene": str(source),
        "source_views": selected, "fit_image_names": fit, "heldout_image_names": heldout, "test_image_names": test,
        "teacher_meta_sha256": file_fingerprint(source / "semantic_meta.npz"),
        "teacher_reference_verified": reference_verified,
        "importance_policy": str(meta["importance_policy"].item()),
        "importance_policy_fingerprint": str(meta.get("importance_policy_fingerprint", "unrecorded")),
        "importance_used_for_bank_ranking": False,
        "model_semantic_reference_sha256": file_fingerprint(model / "semantic_space_reference.json")
        if (model / "semantic_space_reference.json").is_file() else None,
        "source_records": sources, "sampling": vars(args),
        "view_selection": "deterministic_uniform_sorted_training_names",
        "region_selection": "per_level_confidence_round_robin_no_text_or_GT",
        "pixel_selection": "deterministic_uniform_valid_mask_pixels",
        "mask_pooling": "original_overlapping_SAM_masks_not_exclusive_region_map",
        "elapsed_seconds": time.monotonic() - started,
        "quality_status": "unvalidated; short-smoke affinity verifies plumbing only",
    }
    bank = records_to_bank(rows, metadata)
    validate_bank(bank, signature)
    save_bank(output, bank)
    print(json.dumps({"path": str(output), "records": len(rows), "source_views": len(selected),
                      "bytes": output.stat().st_size, "sha256": file_fingerprint(output)}, indent=2))


if __name__ == "__main__":
    main()
