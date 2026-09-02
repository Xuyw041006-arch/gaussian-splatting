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
from utils.general_utils import safe_state


def freeze_gaussians(gaussians):
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        getattr(gaussians, name).requires_grad_(False)


@lru_cache(maxsize=32)
def load_map(path):
    with np.load(path) as data:
        features = torch.from_numpy(data["features"].astype(np.float32))
        valid = torch.from_numpy(data["valid"].astype(bool))
    return features, valid


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
        target, valid = load_map(str(map_path))
        target = target.cuda()
        valid = valid.cuda()
        original_size = (camera.image_height, camera.image_width)
        camera.image_height, camera.image_width = target.shape[-2:]

        chunk_losses = []
        encoded = torch.sigmoid(logits)
        for start in range(0, dimensions, 3):
            stop = min(start + 3, dimensions)
            colors = torch.zeros((encoded.shape[0], 3), device="cuda")
            colors[:, :stop - start] = encoded[:, start:stop]
            rendered = render(
                camera, gaussians, pipeline, background, override_color=colors
            )["render"][:stop - start]
            error = torch.abs(rendered - target[start:stop]).mean(dim=0)
            if valid.any():
                chunk_losses.append(error[valid].mean())
        camera.image_height, camera.image_width = original_size
        if not chunk_losses:
            raise RuntimeError(f"Semantic map has no valid pixels: {map_path}")

        loss = torch.stack(chunk_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema = 0.4 * loss.item() + 0.6 * ema
        if step % 10 == 0:
            progress.set_postfix(loss=f"{ema:.5f}")
        if args.save_every > 0 and step % args.save_every == 0:
            save_artifact(output_path, logits, scene.loaded_iter, meta)

    save_artifact(output_path, logits, scene.loaded_iter, meta)
    print(f"Saved semantic Gaussians: {output_path}")


if __name__ == "__main__":
    main()
