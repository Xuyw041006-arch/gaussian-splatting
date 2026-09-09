"""Runtime for joint RGB reconstruction and hierarchical semantic distillation."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from gaussian_renderer import render
from semantic.curriculum import cosine_ramp, curriculum_phase
from semantic.v5 import (
    affinity_prefix_dimensions, decoded_clip_cosine_loss,
    hierarchical_affinity_loss, normalize_affinity_groups,
    normalize_rendered_features, region_balance_weights,
)
from semantic.joint import (
    symmetric_importance_ema,
    ScaleGate,
    boundary_alignment_loss,
    granularity_for_step,
    load_importance_tiers,
    load_joint_map,
    local_semantic_consistency,
    project_tiers_to_gaussians,
    region_boundaries,
    region_contrastive_loss,
    semantic_chunk_indices,
    select_granularity,
    select_region_ids,
    tier_weights,
)


@lru_cache(maxsize=256)
def load_v5_importance_observations(path):
    """Read lightweight tier evidence only; unknown pixels are not observations.

    Loading NPZ members separately avoids decompressing CLIP feature tensors.
    NaNs are intentionally retained by the symmetric EMA as no-update markers.
    """
    with np.load(path) as data:
        if "importance" not in data.files or "importance_known" not in data.files:
            raise ValueError("V5 requires competitive importance_known evidence maps; regenerate in a new scene directory")
        tiers = data["importance"].astype(np.float32)
        evidence = data["importance_known"]
        if not np.isin(evidence, [0, 1]).all():
            raise ValueError("V5 importance_known must be a binary evidence mask")
        known = evidence.astype(bool)
    if tiers.ndim != 2 or known.shape != tiers.shape:
        raise ValueError("V5 importance tiers and evidence mask must have the same 2D shape")
    if not np.isin(tiers, [0, 1, 2]).all():
        raise ValueError("V5 importance tiers must be 0, 1, or 2")
    return np.where(known, tiers, np.nan).astype(np.float32)


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
        self.v5 = getattr(args, "semantic_protocol", "legacy") == "v5"
        if self.v5 and (
            int(self.meta.get("teacher_preprocessing_version", 0)) != 2
            or str(self.meta.get("hierarchy_method", "")) != "containment"
            or str(self.meta.get("prototype_mode", "")) != "off"
            or str(self.meta.get("importance_policy", "")) != "competitive_v1"
        ):
            raise ValueError("V5 requires a new containment teacher with prototype_mode=off and competitive_v1 importance; do not reuse legacy maps")
        self.affinity_dimensions = int(getattr(args, "affinity_dimensions", 16)) if self.v5 else 0
        already_initialized = gaussians.has_joint_semantics
        gaussians.setup_joint_semantics(
            self.dimensions + self.affinity_dimensions, args.semantic_lr,
        )
        if self.v5 and not already_initialized:
            # A separate raw signed field shares only lifecycle/optimizer storage
            # with language logits, not channels or losses. Random initialization
            # avoids zero-vector symmetry under cosine affinity supervision.
            with torch.no_grad():
                gaussians._semantic_features[:, self.dimensions:].normal_(0, 0.1)
        self.scale_gate = ScaleGate(self.dimensions).cuda()
        self.gate_optimizer = torch.optim.Adam(
            self.scale_gate.parameters(), lr=args.scale_gate_lr
        )
        self.background = torch.zeros(3, dtype=torch.float32, device="cuda")
        self.feature_min = torch.from_numpy(
            self.meta["feature_min"].astype(np.float32)
        ).cuda()
        self.feature_range = torch.from_numpy(
            (self.meta["feature_max"] - self.meta["feature_min"]).astype(np.float32)
        ).cuda()
        if self.v5:
            self.pca_components = torch.from_numpy(self.meta["pca_components"].astype(np.float32)).cuda()
            self.pca_mean = torch.from_numpy(self.meta["pca_mean"].astype(np.float32)).cuda()
            print(
                "Semantic protocol v5: independent language/affinity fields; "
                f"semantic geometry gradients={bool(args.semantic_geometry_grad)}. "
                "Legacy scale gate, prototype pull and language-instance contrastive "
                "losses are disabled."
            )
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
        path = self.map_path(camera) if self.v5 else self.importance_path(camera)
        if not path.is_file():
            return
        tiers = (
            torch.from_numpy(load_v5_importance_observations(str(path)))
            if self.v5 else load_importance_tiers(str(path))
        ).cuda(non_blocking=True)
        indices, observations = project_tiers_to_gaussians(
            self.gaussians.get_xyz, camera, tiers, visible_indices
        )
        if self.v5:
            # Uncovered/ambiguous pixels carry no semantic evidence. In-frame
            # position is not enough to justify changing the persistent tier.
            known = torch.isfinite(observations)
            indices, observations = indices[known], observations[known]
            if not indices.numel():
                return
            previous = self.gaussians.importance_score[indices]
            self.gaussians.importance_score[indices] = symmetric_importance_ema(
                previous, observations.reshape(-1).to(previous), self.args.importance_ema,
            )
        else:
            self.gaussians.update_importance_score(indices, observations, self.args.importance_ema)

    def compute(self, camera, iteration, validation=False):
        """Distill sampled training channels or evaluate all channels at middle scale.

        Validation uses no stochastic contrastive/spatial sampling, so the same
        model and camera have the same data/cross-view/boundary metrics at any step.
        The caller should run validation under torch.no_grad().
        """
        if self.v5:
            return self._compute_v5(camera, iteration, validation)
        path = self.map_path(camera)
        if iteration < self.args.semantic_start or not path.is_file():
            return None
        supervision = load_joint_map(str(path))
        level = 1 if validation else granularity_for_step(iteration)
        target, valid, confidence, prototype_ids = select_granularity(
            supervision, level
        )
        target = target.cuda(non_blocking=True)
        valid = valid.cuda(non_blocking=True)
        confidence = confidence.cuda(non_blocking=True)
        tiers = supervision["importance"].cuda(non_blocking=True)
        detail_weight = supervision["detail_weight"].cuda(non_blocking=True)
        prototype_ids = prototype_ids.cuda(non_blocking=True)
        region_ids = select_region_ids(supervision, level).cuda(non_blocking=True)
        boundary = region_boundaries(region_ids, valid)
        curriculum_weight = cosine_ramp(
            iteration, self.args.semantic_start,
            self.args.semantic_ramp_iterations,
        )

        height, width = target.shape[-2:]
        original_size = (camera.image_height, camera.image_width)
        camera.image_height, camera.image_width = height, width
        try:
            features = self.gaussians.get_semantic_features * self.scale_gate(level)
            selected_chunks = semantic_chunk_indices(
                self.dimensions, self.args.semantic_chunks_per_step,
                iteration, validation=validation,
            )

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
            boundary_losses = []
            contrastive_features = []
            chunk_widths = []
            cross_view_widths = []
            packages = []
            use_contrastive = (
                not validation and self.args.semantic_contrastive_weight > 0
                and iteration % self.args.semantic_contrastive_every == 0
            )
            for chunk in selected_chunks:
                start = 3 * chunk
                stop = min(start + 3, self.dimensions)
                chunk_widths.append(stop - start)
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
                boundary_losses.append(boundary_alignment_loss(
                    prediction, target[start:stop], boundary, active, weights
                ))
                if use_contrastive:
                    # Decode min/max storage before cosine similarity: its
                    # positive offset otherwise makes unrelated regions similar.
                    # Use all channels already rendered in this step jointly;
                    # do not force every three-channel slice to separate regions.
                    contrastive_features.append(
                        prediction * self.feature_range[start:stop, None, None]
                        + self.feature_min[start:stop, None, None]
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
                        cross_view_widths.append(stop - start)
                packages.append(semantic_package)

            def channel_mean(values, widths):
                widths = prediction.new_tensor(widths)
                return (torch.stack(values) * widths).sum() / widths.sum()

            # The last chunk has two channels for a 32-D field. Weight by width
            # so validation is a true mean over all 32 dimensions, not 11 groups.
            data_loss = channel_mean(chunk_losses, chunk_widths)
            cross_view_loss = (
                channel_mean(cross_view_losses, cross_view_widths)
                if cross_view_losses else prediction.new_zeros(())
            )
            boundary_loss = channel_mean(boundary_losses, chunk_widths)
            contrastive_loss = (
                region_contrastive_loss(
                    torch.cat(contrastive_features, dim=0), region_ids, active,
                    self.args.semantic_contrastive_samples, weights,
                ) if contrastive_features else prediction.new_zeros(())
            )

            spatial_loss = prediction.new_zeros(())
            if not validation and iteration % self.args.semantic_spatial_every == 0:
                spatial_loss = local_semantic_consistency(
                    self.gaussians, self.args.semantic_spatial_samples,
                    self.args.semantic_edge_sigma,
                )
            raw_loss = (
                self.args.semantic_weight * data_loss
                + self.args.semantic_cross_view_weight * cross_view_loss
                + self.args.semantic_boundary_weight * boundary_loss
                + self.args.semantic_contrastive_weight * contrastive_loss
                + self.args.semantic_spatial_weight
                * self.args.semantic_spatial_every * spatial_loss
            )
            loss = float(curriculum_weight) * raw_loss
            return {
                "loss": loss,
                "data_loss": data_loss.detach(),
                "spatial_loss": spatial_loss.detach(),
                "cross_view_loss": cross_view_loss.detach(),
                "boundary_loss": boundary_loss.detach(),
                "contrastive_loss": contrastive_loss.detach(),
                "curriculum_weight": float(curriculum_weight),
                "phase": curriculum_phase(
                    iteration, self.args.semantic_start,
                    self.args.semantic_ramp_iterations,
                ),
                "packages": packages,
                "level": level,
                "chunks": selected_chunks,
                "evaluated_dimensions": sum(chunk_widths),
                "validation": bool(validation),
            }
        finally:
            camera.image_height, camera.image_width = original_size

    def _compute_v5(self, camera, iteration, validation=False):
        """Distill language and SAM hierarchy without contradictory objectives.

        RGB geometry remains trainable by RGB loss. By default the noisy 2-D
        semantic teacher updates feature fields only; geometry-coupled semantics
        is an explicit ablation with a differentiable alpha denominator.
        """
        path = self.map_path(camera)
        if iteration < self.args.semantic_start or not path.is_file():
            return None
        supervision = load_joint_map(str(path))
        target = supervision["features"].cuda(non_blocking=True)
        valid = supervision["valid"].cuda(non_blocking=True)
        confidence = supervision["confidence"].cuda(non_blocking=True)
        region_ids = supervision["region_ids"].cuda(non_blocking=True)
        tiers = supervision["importance"].cuda(non_blocking=True)
        detail = supervision["detail_weight"].cuda(non_blocking=True)
        weights = confidence * tier_weights(tiers, self.args.semantic_tier_weights) * detail
        weights *= region_balance_weights(
            region_ids, valid, self.args.semantic_region_balance_power,
            self.args.semantic_region_balance_cap,
        )
        hierarchy = supervision["hierarchy_region_ids"]
        hierarchy = (
            hierarchy.cuda(non_blocking=True) if hierarchy is not None
            else region_ids.unsqueeze(0).expand(3, -1, -1)
        )
        # Full-vector cosine is periodic for training and deterministic for val.
        use_cosine = self.args.semantic_clip_cosine_weight > 0 and (
            validation or iteration % self.args.semantic_clip_cosine_every == 0
        )
        use_affinity = not validation and self.args.affinity_weight > 0 and (
            iteration % self.args.affinity_every == 0
        )
        selected_chunks = semantic_chunk_indices(
            self.dimensions, self.args.semantic_chunks_per_step, iteration,
            validation=validation or use_cosine,
        )
        language = torch.sigmoid(self.gaussians._semantic_features[:, :self.dimensions])
        detach_geometry = not self.args.semantic_geometry_grad
        render_options = {"detach_geometry": detach_geometry, "clamp_output": False}
        original_size = camera.image_height, camera.image_width
        camera.image_height, camera.image_width = target.shape[-2:]
        try:
            alpha = render(
                camera, self.gaussians, self.pipeline, self.background,
                override_color=torch.ones_like(language[:, :3]), **render_options,
            )["render"][:1]
            coverage = alpha[0].detach() >= max(float(self.args.semantic_min_alpha), 1e-4)
            active = valid & coverage & (weights > 0)
            if not active.any():
                return None
            boundary = region_boundaries(region_ids, valid)
            data_losses, boundary_losses, widths, predictions, packages = [], [], [], [], []
            for chunk in selected_chunks:
                start, stop = 3 * chunk, min(3 * chunk + 3, self.dimensions)
                width = stop - start
                colors = torch.nn.functional.pad(language[:, start:stop], (0, 3 - width))
                package = render(
                    camera, self.gaussians, self.pipeline, self.background,
                    override_color=colors, **render_options,
                )
                prediction = normalize_rendered_features(package["render"][:width], alpha)
                error = (prediction - target[start:stop]).abs().mean(dim=0)
                data_losses.append((error[active] * weights[active]).sum() / weights[active].sum().clamp_min(1e-8))
                boundary_losses.append(boundary_alignment_loss(
                    prediction, target[start:stop], boundary, active, weights,
                ))
                widths.append(width)
                if use_cosine:
                    predictions.append(prediction)
                packages.append(package)
            channel_weights = prediction.new_tensor(widths)
            data_loss = (torch.stack(data_losses) * channel_weights).sum() / channel_weights.sum()
            boundary_loss = (torch.stack(boundary_losses) * channel_weights).sum() / channel_weights.sum()
            zero = data_loss.new_zeros(())
            cosine_loss = decoded_clip_cosine_loss(
                torch.cat(predictions), target, active, weights,
                self.feature_min, self.feature_range, self.pca_components, self.pca_mean,
                deterministic=validation,
            ) if use_cosine else zero
            affinity_loss, affinity_stats = zero, {"pairs": 0, "conflicting_pairs": 0}
            if use_affinity:
                affinity = normalize_affinity_groups(
                    self.gaussians._semantic_features[:, self.dimensions:],
                )
                affinity_images = []
                for start in range(0, self.affinity_dimensions, 3):
                    width = min(3, self.affinity_dimensions - start)
                    package = render(
                        camera, self.gaussians, self.pipeline, self.background,
                        override_color=torch.nn.functional.pad(affinity[:, start:start + width], (0, 3 - width)),
                        **render_options,
                    )
                    affinity_images.append(normalize_rendered_features(package["render"][:width], alpha))
                    packages.append(package)
                affinity_loss, affinity_stats = hierarchical_affinity_loss(
                    torch.cat(affinity_images), hierarchy, coverage,
                    # Do not double-apply inverse-area balancing: affinity
                    # sampling adds its own hierarchy-aware area correction.
                    weights=confidence * tier_weights(tiers, self.args.semantic_tier_weights) * detail,
                    samples=self.args.affinity_samples,
                )
            curriculum_weight = cosine_ramp(
                iteration, self.args.semantic_start, self.args.semantic_ramp_iterations,
            )
            cosine_cadence = 1 if validation else self.args.semantic_clip_cosine_every
            loss = float(curriculum_weight) * (
                self.args.semantic_weight * data_loss
                + self.args.semantic_boundary_weight * boundary_loss
                + self.args.semantic_clip_cosine_weight * cosine_cadence * cosine_loss
                + self.args.affinity_weight * self.args.affinity_every * affinity_loss
            )
            return {
                "loss": loss, "data_loss": data_loss.detach(),
                "boundary_loss": boundary_loss.detach(), "clip_cosine_loss": cosine_loss.detach(),
                "affinity_loss": affinity_loss.detach(), "affinity_stats": affinity_stats,
                "spatial_loss": zero, "cross_view_loss": zero, "contrastive_loss": zero,
                "curriculum_weight": float(curriculum_weight),
                "phase": curriculum_phase(iteration, self.args.semantic_start, self.args.semantic_ramp_iterations),
                "packages": packages, "level": "language_base+independent_hierarchy",
                "chunks": selected_chunks, "evaluated_dimensions": sum(widths),
                "validation": bool(validation), "semantic_protocol": "v5",
            }
        finally:
            camera.image_height, camera.image_width = original_size

    def step(self):
        self.gate_optimizer.step()
        self.gate_optimizer.zero_grad(set_to_none=True)

    def checkpoint_state(self):
        """State not owned by GaussianModel and needed for exact continuation."""
        return {
            "semantic_protocol": "v5" if self.v5 else "legacy",
            "language_dimensions": self.dimensions,
            "affinity_dimensions": self.affinity_dimensions,
            "scale_gate": self.scale_gate.state_dict(),
            "gate_optimizer": self.gate_optimizer.state_dict(),
        }

    def restore_checkpoint_state(self, state):
        if not state:
            return
        expected = "v5" if self.v5 else "legacy"
        if state.get("semantic_protocol", "legacy") != expected:
            raise ValueError("Cannot resume a different semantic protocol; use a separate v5 run directory")
        if self.v5 and (
            state.get("language_dimensions") != self.dimensions
            or state.get("affinity_dimensions") != self.affinity_dimensions
        ):
            raise ValueError("V5 language/affinity channel layout differs from checkpoint")
        if "scale_gate" in state:
            self.scale_gate.load_state_dict(state["scale_gate"])
        if "gate_optimizer" in state:
            self.gate_optimizer.load_state_dict(state["gate_optimizer"])
        print(f"Restored joint semantic supervisor checkpoint ({expected})")

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
            "features": torch.sigmoid(self.gaussians._semantic_features[:, :self.dimensions]).detach().half().cpu(),
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
            "semantic_start": int(self.args.semantic_start),
            "semantic_ramp_iterations": int(self.args.semantic_ramp_iterations),
            "semantic_boundary_weight": float(self.args.semantic_boundary_weight),
            "semantic_contrastive_weight": float(
                self.args.semantic_contrastive_weight
            ),
        }
        if self.v5:
            artifact.update({
                "version": 5, "semantic_protocol": "v5",
                "affinity_features": normalize_affinity_groups(self.gaussians._semantic_features[:, self.dimensions:]).detach().half().cpu(),
                "affinity_prefix_dimensions": affinity_prefix_dimensions(self.affinity_dimensions),
                "affinity_level_order": ("coarse", "middle", "fine"),
                "semantic_geometry_grad": bool(self.args.semantic_geometry_grad),
                "semantic_clip_cosine_weight": float(self.args.semantic_clip_cosine_weight),
                "semantic_clip_cosine_every": int(self.args.semantic_clip_cosine_every),
                "affinity_weight": float(self.args.affinity_weight),
                "affinity_every": int(self.args.affinity_every),
                "semantic_cross_view_weight": 0.0,
                "semantic_contrastive_weight": 0.0,
                "language_target": "raw_base_pca_no_global_prototype_pull",
            })
            artifact.pop("scale_gate", None)
        torch.save(artifact, output)
        return output
