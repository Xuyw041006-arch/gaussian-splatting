"""Runtime for joint RGB reconstruction and hierarchical semantic distillation."""

import os
from pathlib import Path

import numpy as np
import torch

from gaussian_renderer import render
from semantic.joint import (
    ScaleGate,
    granularity_for_step,
    load_importance_tiers,
    load_joint_map,
    local_semantic_consistency,
    project_tiers_to_gaussians,
    select_granularity,
    tier_weights,
)


class JointSemanticSupervisor:
    def __init__(self, dataset, gaussians, pipeline, args):
        self.dataset = dataset
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.args = args
        self.semantic_dir = Path(
            args.semantic_dir or os.path.join(dataset.source_path, "semantic_maps")
        )
        self.importance_dir = Path(
            args.importance_mask_dir
            or os.path.join(dataset.source_path, "importance_masks")
        )
        meta_path = Path(dataset.source_path) / "semantic_meta.npz"
        if not self.semantic_dir.is_dir() or not meta_path.is_file():
            raise FileNotFoundError(
                "Joint training requires preprocess_semantics.py outputs"
            )
        with np.load(meta_path) as loaded:
            self.meta = {key: loaded[key].copy() for key in loaded.files}
        self.dimensions = int(self.meta["pca_components"].shape[0])
        gaussians.setup_joint_semantics(self.dimensions, args.semantic_lr)
        self.scale_gate = ScaleGate(self.dimensions).cuda()
        self.gate_optimizer = torch.optim.Adam(
            self.scale_gate.parameters(), lr=args.scale_gate_lr
        )
        self.background = torch.zeros(3, dtype=torch.float32, device="cuda")
        self.prototype_features = None
        if "prototype_features" in self.meta:
            self.prototype_features = torch.from_numpy(
                self.meta["prototype_features"].astype(np.float32)
            ).cuda()

    def map_path(self, camera):
        return self.semantic_dir / f"{Path(camera.image_name).stem}.npz"

    def importance_path(self, camera):
        return self.importance_dir / f"{Path(camera.image_name).stem}.png"

    @torch.no_grad()
    def observe_importance(self, camera, visible_indices):
        """Fuse tier evidence from every RGB iteration, including semantic warmup."""
        path = self.importance_path(camera)
        if not path.is_file():
            return
        tiers = load_importance_tiers(str(path)).cuda(non_blocking=True)
        indices, observations = project_tiers_to_gaussians(
            self.gaussians.get_xyz, camera, tiers, visible_indices
        )
        self.gaussians.update_importance_score(
            indices, observations, self.args.importance_ema
        )

    def compute(self, camera, iteration):
        path = self.map_path(camera)
        if iteration < self.args.semantic_start or not path.is_file():
            return None
        supervision = load_joint_map(str(path))
        level = granularity_for_step(iteration)
        target, valid, confidence, prototype_ids = select_granularity(
            supervision, level
        )
        target = target.cuda(non_blocking=True)
        valid = valid.cuda(non_blocking=True)
        confidence = confidence.cuda(non_blocking=True)
        tiers = supervision["importance"].cuda(non_blocking=True)
        detail_weight = supervision["detail_weight"].cuda(non_blocking=True)
        prototype_ids = prototype_ids.cuda(non_blocking=True)

        height, width = target.shape[-2:]
        original_size = (camera.image_height, camera.image_width)
        camera.image_height, camera.image_width = height, width
        try:
            features = self.gaussians.get_semantic_features * self.scale_gate(level)
            chunks = (self.dimensions + 2) // 3

            with torch.no_grad():
                alpha = render(
                    camera, self.gaussians, self.pipeline, self.background,
                    override_color=torch.ones(
                        (features.shape[0], 3), dtype=features.dtype, device="cuda"
                    ),
                )["render"][:1].clamp(0, 1)
            active = valid & (alpha[0] >= self.args.semantic_min_alpha)
            if not active.any():
                return None
            weights = (
                confidence
                * tier_weights(tiers, self.args.semantic_tier_weights)
                * detail_weight
            )
            chunk_losses = []
            cross_view_losses = []
            packages = []
            first_chunk = (iteration * self.args.semantic_chunks_per_step) % chunks
            for offset in range(min(self.args.semantic_chunks_per_step, chunks)):
                chunk = (first_chunk + offset) % chunks
                start = 3 * chunk
                stop = min(start + 3, self.dimensions)
                colors = torch.zeros((features.shape[0], 3), device="cuda")
                colors[:, :stop - start] = features[:, start:stop]
                semantic_package = render(
                    camera, self.gaussians, self.pipeline, self.background,
                    override_color=colors,
                )
                prediction = semantic_package["render"][:stop - start]
                prediction = prediction / alpha.clamp_min(1e-4)
                error = torch.abs(prediction - target[start:stop]).mean(dim=0)
                chunk_losses.append(
                    (error[active] * weights[active]).sum()
                    / weights[active].sum().clamp_min(1e-8)
                )
                if self.prototype_features is not None:
                    prototype_valid = active & (prototype_ids >= 0)
                    if prototype_valid.any():
                        prototype_target = self.prototype_features[
                            prototype_ids[prototype_valid]
                        ][:, start:stop].T
                        prototype_error = torch.abs(
                            prediction[:, prototype_valid] - prototype_target
                        ).mean(dim=0)
                        prototype_weights = weights[prototype_valid]
                        cross_view_losses.append(
                            (prototype_error * prototype_weights).sum()
                            / prototype_weights.sum().clamp_min(1e-8)
                        )
                packages.append(semantic_package)
            data_loss = torch.stack(chunk_losses).mean()
            cross_view_loss = (
                torch.stack(cross_view_losses).mean()
                if cross_view_losses else prediction.new_zeros(())
            )

            spatial_loss = prediction.new_zeros(())
            if iteration % self.args.semantic_spatial_every == 0:
                spatial_loss = local_semantic_consistency(
                    self.gaussians, self.args.semantic_spatial_samples,
                    self.args.semantic_edge_sigma,
                )
            loss = (
                self.args.semantic_weight * data_loss
                + self.args.semantic_cross_view_weight * cross_view_loss
                + self.args.semantic_spatial_weight
                * self.args.semantic_spatial_every * spatial_loss
            )
            return {
                "loss": loss,
                "data_loss": data_loss.detach(),
                "spatial_loss": spatial_loss.detach(),
                "cross_view_loss": cross_view_loss.detach(),
                "packages": packages,
                "level": level,
                "chunks": [
                    (first_chunk + offset) % chunks
                    for offset in range(min(self.args.semantic_chunks_per_step, chunks))
                ],
            }
        finally:
            camera.image_height, camera.image_width = original_size

    def step(self):
        self.gate_optimizer.step()
        self.gate_optimizer.zero_grad(set_to_none=True)

    def checkpoint_state(self):
        """State not owned by GaussianModel and needed for exact continuation."""
        return {
            "scale_gate": self.scale_gate.state_dict(),
            "gate_optimizer": self.gate_optimizer.state_dict(),
        }

    def restore_checkpoint_state(self, state):
        if not state:
            return
        if "scale_gate" in state:
            self.scale_gate.load_state_dict(state["scale_gate"])
        if "gate_optimizer" in state:
            self.gate_optimizer.load_state_dict(state["gate_optimizer"])
        print("Restored joint semantic scale-gate checkpoint")

    def save(self, iteration):
        output = (
            Path(self.dataset.model_path) / "semantic" / f"iteration_{iteration}"
            / "semantic_features.pt"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "version": 2,
            "training": "joint",
            "scene_iteration": int(iteration),
            "features": self.gaussians.get_semantic_features.detach().half().cpu(),
            "importance_score": self.gaussians.importance_score.detach().half().cpu(),
            "scale_gate": {
                key: value.detach().cpu()
                for key, value in self.scale_gate.state_dict().items()
            },
            "pca_components": torch.from_numpy(self.meta["pca_components"].astype(np.float32)),
            "pca_mean": torch.from_numpy(self.meta["pca_mean"].astype(np.float32)),
            "feature_min": torch.from_numpy(self.meta["feature_min"].astype(np.float32)),
            "feature_max": torch.from_numpy(self.meta["feature_max"].astype(np.float32)),
            "clip_model": str(self.meta["clip_model"].item()),
            "clip_pretrained": str(self.meta["clip_pretrained"].item()),
            "tier_rgb_weights": tuple(self.args.rgb_tier_weights),
            "tier_semantic_weights": tuple(self.args.semantic_tier_weights),
            "tier_sh_degrees": tuple(self.args.tier_sh_degrees),
            "semantic_cross_view_weight": float(
                self.args.semantic_cross_view_weight
            ),
            "semantic_edge_sigma": float(self.args.semantic_edge_sigma),
        }
        torch.save(artifact, output)
        return output
