"""Render a smooth turntable video from a trained 3D Gaussian model."""

import math
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix, getWorld2View2

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SEPARATE_SH = True
except ImportError:
    SEPARATE_SH = False


def normalized(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def axis_vector(name):
    value = np.zeros(3, dtype=np.float32)
    value[{"x": 0, "y": 1, "z": 2}[name]] = 1.0
    return value


def make_orbit_positions(camera_centers, target, frames, up, radius_scale):
    relative = camera_centers - target[None]
    heights = relative @ up
    planar = relative - heights[:, None] * up[None]
    planar_norms = np.linalg.norm(planar, axis=1)
    valid = planar_norms > 1e-5
    if not np.any(valid):
        raise ValueError("Camera centers do not define an orbit plane")

    first = int(np.flatnonzero(valid)[0])
    basis_u = planar[first] / planar_norms[first]
    basis_v = normalized(np.cross(up, basis_u))
    radius = float(np.median(planar_norms[valid])) * radius_scale
    height = float(np.median(heights))

    positions = []
    for angle in np.linspace(0.0, 2.0 * math.pi, frames, endpoint=False):
        radial = math.cos(angle) * basis_u + math.sin(angle) * basis_v
        positions.append(target + radius * radial + height * up)
    return np.asarray(positions, dtype=np.float32)


def make_camera(reference, position, target, up):
    forward = normalized(target - position)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(forward @ fallback)) > 0.95:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(forward, fallback)
    right = normalized(right)
    down = normalized(np.cross(forward, right))

    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :3] = np.column_stack([right, down, forward])
    camera_to_world[:3, 3] = position
    world_to_camera = np.linalg.inv(camera_to_world)
    rotation = world_to_camera[:3, :3].T
    translation = world_to_camera[:3, 3]

    world_view = torch.tensor(
        getWorld2View2(rotation, translation), dtype=torch.float32, device="cuda"
    ).transpose(0, 1)
    projection = getProjectionMatrix(
        znear=reference.znear,
        zfar=reference.zfar,
        fovX=reference.FoVx,
        fovY=reference.FoVy,
    ).transpose(0, 1).cuda()
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(
        reference.image_width,
        reference.image_height,
        reference.FoVy,
        reference.FoVx,
        reference.znear,
        reference.zfar,
        world_view,
        full_projection,
    )


def tensor_to_image(rendering):
    pixels = (
        rendering.detach().clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def comparison_frame(original, edited=None):
    images = [original] if edited is None else [original, edited]
    labels = ["Rendered 3DGS"] if edited is None else ["Original 3DGS", "After text deletion"]
    width, height = images[0].size
    title_height = 32
    canvas = Image.new("RGB", (width * len(images), height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (image, label) in enumerate(zip(images, labels)):
        left = index * width
        canvas.paste(image, (left, title_height))
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text((left + (width - text_width) / 2, 10), label, fill="black", font=font)
    return canvas


def encode_video(frame_directory, output, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the orbit video")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_directory / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")


def render_orbit(dataset, pipeline, iteration, edited_model, output, frames, fps, up_axis, radius_scale):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        reference_views = scene.getTestCameras() or scene.getTrainCameras()
        if not reference_views:
            raise RuntimeError("The scene has no cameras")

        edited_gaussians = None
        if edited_model:
            edited_ply = (
                Path(edited_model)
                / "point_cloud"
                / f"iteration_{scene.loaded_iter}"
                / "point_cloud.ply"
            )
            if not edited_ply.is_file():
                raise FileNotFoundError(edited_ply)
            edited_gaussians = GaussianModel(dataset.sh_degree)
            edited_gaussians.load_ply(str(edited_ply), dataset.train_test_exp)

        all_views = scene.getTrainCameras() + scene.getTestCameras()
        centers = np.stack([view.camera_center.detach().cpu().numpy() for view in all_views])
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        target = np.median(xyz, axis=0)
        up = axis_vector(up_axis)
        positions = make_orbit_positions(centers, target, frames, up, radius_scale)
        background = torch.tensor(
            [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame_directory = output.with_suffix("").with_name(output.stem + "_frames")
        frame_directory.mkdir(parents=True, exist_ok=True)
        preview = output.with_name(output.stem + "_preview.png")

        for index, position in enumerate(tqdm(positions, desc="Rendering orbit")):
            camera = make_camera(reference_views[0], position, target, up)
            original = tensor_to_image(
                render(camera, gaussians, pipeline, background, separate_sh=SEPARATE_SH)["render"]
            )
            edited = None
            if edited_gaussians is not None:
                edited = tensor_to_image(
                    render(
                        camera,
                        edited_gaussians,
                        pipeline,
                        background,
                        separate_sh=SEPARATE_SH,
                    )["render"]
                )
            frame = comparison_frame(original, edited)
            frame.save(frame_directory / f"{index:05d}.png")
            if index == 0:
                frame.save(preview)

        encode_video(frame_directory, output, fps)
        shutil.rmtree(frame_directory)
        print(f"Orbit video: {output}")
        print(f"Preview: {preview}")
        return output, preview


if __name__ == "__main__":
    parser = ArgumentParser(description="Render a 360-degree 3DGS turntable video")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--edited_model", default="")
    parser.add_argument("--output", default="orbit.mp4")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--up_axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if args.frames < 2 or args.fps < 1 or args.radius_scale <= 0:
        parser.error("frames >= 2, fps >= 1 and radius_scale > 0 are required")

    safe_state(args.quiet)
    render_orbit(
        model.extract(args),
        pipeline.extract(args),
        args.iteration,
        args.edited_model,
        args.output,
        args.frames,
        args.fps,
        args.up_axis,
        args.radius_scale,
    )
