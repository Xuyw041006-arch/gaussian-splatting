#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.importance_utils import (
    importance_to_tiers,
    load_importance_mask,
    weighted_l1,
    weighted_tier_l1,
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from, importance_mask_dir="", foreground_weight=4.0,
             background_weight=0.25, joint_args=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    joint = None
    if joint_args is not None and joint_args.joint_semantics:
        if opt.optimizer_type == "sparse_adam":
            raise ValueError("Joint semantic fields currently require the default Adam optimizer")
        from semantic.joint_trainer import JointSemanticSupervisor
        joint = JointSemanticSupervisor(dataset, gaussians, pipe, joint_args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    importance_cache = {}

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if joint is not None:
            joint.observe_importance(viewpoint_cam, visibility_filter)

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        importance_mask = load_importance_mask(
            importance_mask_dir, viewpoint_cam.image_name, image.shape[-2:],
            image.device, importance_cache
        )
        if importance_mask is None:
            Ll1 = l1_loss(image, gt_image)
        elif joint is not None:
            Ll1 = weighted_tier_l1(
                image, gt_image, importance_to_tiers(importance_mask),
                joint_args.rgb_tier_weights,
            )
        else:
            Ll1 = weighted_l1(
                image, gt_image, importance_mask,
                foreground_weight=foreground_weight,
                background_weight=background_weight,
            )
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        joint_result = joint.compute(viewpoint_cam, iteration) if joint is not None else None
        if joint_result is not None:
            loss += joint_result["loss"]

        loss.backward()
        if joint is not None:
            gaussians.mask_sh_gradients(*joint_args.tier_sh_degrees)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if joint is not None:
                    semantic_path = joint.save(iteration)
                    print(f"[ITER {iteration}] Saved joint semantics: {semantic_path}")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if joint_result is not None:
                    for semantic_package in joint_result["packages"]:
                        gaussians.add_densification_stats(
                            semantic_package["viewspace_points"],
                            semantic_package["visibility_filter"],
                        )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                        size_threshold, radii,
                        joint_args.tier_densify_multipliers if joint is not None else None,
                        joint_args.tier_opacity_multipliers if joint is not None else None,
                    )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if joint is not None:
                    joint.step()
                    gaussians.enforce_sh_capacity(*joint_args.tier_sh_degrees)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
        "--importance_mask_dir", type=str, default="",
        help="Optional grayscale masks named after input images"
    )
    parser.add_argument("--foreground_weight", type=float, default=4.0)
    parser.add_argument("--background_weight", type=float, default=0.25)
    parser.add_argument("--joint_semantics", action="store_true")
    parser.add_argument("--semantic_dir", default="")
    parser.add_argument("--semantic_start", type=int, default=1000)
    parser.add_argument("--semantic_weight", type=float, default=0.15)
    parser.add_argument("--semantic_lr", type=float, default=0.005)
    parser.add_argument("--scale_gate_lr", type=float, default=0.001)
    parser.add_argument("--semantic_min_alpha", type=float, default=0.05)
    parser.add_argument("--semantic_spatial_weight", type=float, default=0.02)
    parser.add_argument("--semantic_spatial_every", type=int, default=8)
    parser.add_argument("--semantic_spatial_samples", type=int, default=512)
    parser.add_argument("--semantic_chunks_per_step", type=int, default=3)
    parser.add_argument("--importance_ema", type=float, default=0.90)
    parser.add_argument(
        "--rgb_tier_weights", nargs=3, type=float, default=(0.35, 1.0, 4.0),
        metavar=("BACKGROUND", "NORMAL", "IMPORTANT"),
    )
    parser.add_argument(
        "--semantic_tier_weights", nargs=3, type=float, default=(0.15, 1.0, 4.0),
        metavar=("BACKGROUND", "NORMAL", "IMPORTANT"),
    )
    parser.add_argument(
        "--tier_densify_multipliers", nargs=3, type=float,
        default=(1.80, 1.0, 0.55),
        metavar=("BACKGROUND", "NORMAL", "IMPORTANT"),
    )
    parser.add_argument(
        "--tier_opacity_multipliers", nargs=3, type=float,
        default=(2.0, 1.0, 0.5),
        metavar=("BACKGROUND", "NORMAL", "IMPORTANT"),
    )
    parser.add_argument(
        "--tier_sh_degrees", nargs=3, type=int, default=(1, 3, 5),
        metavar=("BACKGROUND", "NORMAL", "IMPORTANT"),
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.importance_mask_dir:
        args.importance_mask_dir = os.path.abspath(args.importance_mask_dir)
        if not os.path.isdir(args.importance_mask_dir):
            parser.error(f"Importance mask directory does not exist: {args.importance_mask_dir}")
    if args.foreground_weight <= 0 or args.background_weight <= 0:
        parser.error("Importance weights must be positive")
    if args.joint_semantics:
        positive = (
            list(args.rgb_tier_weights) + list(args.semantic_tier_weights)
            + list(args.tier_densify_multipliers)
            + list(args.tier_opacity_multipliers)
            + [args.semantic_weight, args.semantic_lr, args.scale_gate_lr]
        )
        if min(positive) <= 0:
            parser.error("Joint semantic weights and learning rates must be positive")
        if (
            args.semantic_start < 0 or args.semantic_spatial_every < 1
            or args.semantic_chunks_per_step < 1
        ):
            parser.error("Semantic start/every values are invalid")
        if not 0 <= args.importance_ema < 1 or not 0 <= args.semantic_min_alpha < 1:
            parser.error("EMA and minimum alpha must be in [0, 1)")
        if max(args.tier_sh_degrees) > args.sh_degree:
            parser.error("--sh_degree must cover every --tier_sh_degrees value")
    training(
        lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from, args.importance_mask_dir, args.foreground_weight,
        args.background_weight, args
    )

    # All done
    print("\nTraining complete.")
