"""Distill 2D SAM+CLIP feature maps into the trained 3D Gaussians."""

import os
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from random import randint

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from semantic.regularization import build_neighbor_graph
from utils.general_utils import safe_state
from utils.sh_utils import SH2RGB


def freeze_gaussians(gaussians):
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        getattr(gaussians, name).requires_grad_(False)


@lru_cache(maxsize=32)
def load_map(path):
    with np.load(path) as data:
        features = torch.from_numpy(data["features"].astype(np.float32))
        valid = torch.from_numpy(data["valid"].astype(bool))
        confidence = torch.from_numpy(
            data["confidence"].astype(np.float32)
            if "confidence" in data.files else np.ones(data["valid"].shape, dtype=np.float32)
        )
    return features, valid, confidence


def save_artifact(path, logits, scene_iteration, meta):
    artifact = {
        "version": 1,
        "scene_iteration": int(scene_iteration),
        "features": torch.sigmoid(logits.detach()).half().cpu(),
        "pca_components": torch.from_numpy(meta["pca_components"].astype(np.float32)),
        "pca_mean": torch.from_numpy(meta["pca_mean"].astype(np.float32)),
        "feature_min": torch.from_numpy(meta["feature_min"].astype(np.float32)),
        "feature_max": torch.from_numpy(meta["feature_max"].astype(np.float32)),
        "clip_model": str(meta["clip_model"].item()),
        "clip_pretrained": str(meta["clip_pretrained"].item()),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(artifact, path)


def main():
    parser = ArgumentParser(description="Train open-vocabulary features on fixed 3D Gaussians")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--semantic_dir", type=str, default="")
    parser.add_argument("--semantic_iterations", type=int, default=5000)
    parser.add_argument("--semantic_lr", type=float, default=0.01)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--spatial_weight", type=float, default=0.02)
    parser.add_argument("--spatial_k", type=int, default=8)
    parser.add_argument("--spatial_samples", type=int, default=4096)
    parser.add_argument("--spatial_color_sigma", type=float, default=0.25)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument(
        "--no_alpha_normalization", action="store_true",
        help="Disable opacity-normalized semantic feature rendering",
    )
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    if not torch.cuda.is_available():
        parser.error("CUDA is required for semantic Gaussian training")
    dataset = model_params.extract(args)
    semantic_dir = Path(args.semantic_dir or os.path.join(dataset.source_path, "semantic_maps"))
    meta_path = Path(dataset.source_path) / "semantic_meta.npz"
    if not semantic_dir.is_dir() or not meta_path.is_file():
        parser.error("Run preprocess_semantics.py before train_semantics.py")
    if args.semantic_iterations < 1 or args.semantic_lr <= 0:
        parser.error("Semantic iterations and learning rate must be positive")
    if args.spatial_weight < 0 or args.spatial_k < 1 or args.spatial_samples < 1:
        parser.error("Spatial weight must be non-negative; k and samples must be positive")
    if args.spatial_color_sigma <= 0 or not 0 <= args.min_alpha < 1:
        parser.error("Spatial color sigma must be positive and min alpha must be in [0, 1)")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    freeze_gaussians(gaussians)
    with np.load(meta_path) as loaded_meta:
        meta = {key: loaded_meta[key].copy() for key in loaded_meta.files}
    dimensions = int(meta["pca_components"].shape[0])
    logits = torch.nn.Parameter(
        torch.zeros((gaussians.get_xyz.shape[0], dimensions), device="cuda")
    )
    optimizer = torch.optim.Adam([logits], lr=args.semantic_lr)

    neighbor_indices = neighbor_weights = None
    if args.spatial_weight > 0:
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        colors = SH2RGB(gaussians.get_features[:, 0, :]).detach().clamp(0, 1).cpu().numpy()
        neighbor_indices_np, neighbor_weights_np = build_neighbor_graph(
            xyz, colors, args.spatial_k, args.spatial_color_sigma
        )
        neighbor_indices = torch.from_numpy(neighbor_indices_np).cuda()
        neighbor_weights = torch.from_numpy(neighbor_weights_np).cuda()
    cameras = scene.getTrainCameras().copy()
    available = [
        camera for camera in cameras
        if (semantic_dir / f"{Path(camera.image_name).stem}.npz").is_file()
    ]
    if not available:
        parser.error(f"No semantic maps matched cameras in {semantic_dir}")

    pipeline = pipeline_params.extract(args)
    background = torch.zeros(3, device="cuda")
    output_path = os.path.join(
        dataset.model_path, "semantic", f"iteration_{scene.loaded_iter}",
        "semantic_features.pt",
    )
    progress = tqdm(range(1, args.semantic_iterations + 1), desc="Semantic training")
    stack = []
    ema = 0.0
    for step in progress:
        if not stack:
            stack = available.copy()
        camera = stack.pop(randint(0, len(stack) - 1))
        map_path = semantic_dir / f"{Path(camera.image_name).stem}.npz"
        target, valid, confidence = load_map(str(map_path))
        target = target.cuda()
        valid = valid.cuda()
        confidence = confidence.cuda()
        original_size = (camera.image_height, camera.image_width)
        camera.image_height, camera.image_width = target.shape[-2:]

        chunk_losses = []
        encoded = torch.sigmoid(logits)
        alpha = None
        if not args.no_alpha_normalization:
            with torch.no_grad():
                alpha = render(
                    camera, gaussians, pipeline, background,
                    override_color=torch.ones((encoded.shape[0], 3), device="cuda"),
                )["render"][:1].clamp(0, 1)
        for start in range(0, dimensions, 3):
            stop = min(start + 3, dimensions)
            colors = torch.zeros((encoded.shape[0], 3), device="cuda")
            colors[:, :stop - start] = encoded[:, start:stop]
            rendered = render(
                camera, gaussians, pipeline, background, override_color=colors
            )["render"][:stop - start]
            current_valid = valid
            if alpha is not None:
                rendered = rendered / alpha.clamp_min(1e-4)
                current_valid = valid & (alpha[0] >= args.min_alpha)
            error = torch.abs(rendered - target[start:stop]).mean(dim=0)
            if current_valid.any():
                weights = confidence[current_valid]
                chunk_losses.append(
                    (error[current_valid] * weights).sum() / weights.sum().clamp_min(1e-8)
                )
        camera.image_height, camera.image_width = original_size
        if not chunk_losses:
            raise RuntimeError(f"Semantic map has no valid pixels: {map_path}")

        data_loss = torch.stack(chunk_losses).mean()
        spatial_loss = torch.zeros((), device="cuda")
        if neighbor_indices is not None:
            sample_count = min(args.spatial_samples, encoded.shape[0])
            source = torch.randint(0, encoded.shape[0], (sample_count,), device="cuda")
            columns = torch.randint(0, neighbor_indices.shape[1], (sample_count,), device="cuda")
            target_indices = neighbor_indices[source, columns]
            pair_weights = neighbor_weights[source, columns]
            pair_error = torch.nn.functional.smooth_l1_loss(
                encoded[source], encoded[target_indices], reduction="none"
            ).mean(dim=-1)
            spatial_loss = (
                pair_error * pair_weights
            ).sum() / pair_weights.sum().clamp_min(1e-8)
        loss = data_loss + args.spatial_weight * spatial_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema = 0.4 * loss.item() + 0.6 * ema
        if step % 10 == 0:
            progress.set_postfix(
                loss=f"{ema:.5f}", spatial=f"{spatial_loss.item():.5f}"
            )
        if args.save_every > 0 and step % args.save_every == 0:
            save_artifact(output_path, logits, scene.loaded_iter, meta)

    save_artifact(output_path, logits, scene.loaded_iter, meta)
    print(f"Saved semantic Gaussians: {output_path}")


if __name__ == "__main__":
    main()
