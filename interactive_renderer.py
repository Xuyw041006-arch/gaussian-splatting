"""Interactive Colab/Jupyter controls for rasterized 3D Gaussian models."""

import io
import math
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from render_orbit import SEPARATE_SH, axis_vector, make_camera, normalized, tensor_to_image
from scene import Scene


def read_model_config(model_path):
    text = (Path(model_path) / "cfg_args").read_text(encoding="utf-8")
    config = eval(text, {"__builtins__": {}, "Namespace": Namespace})
    config.model_path = str(Path(model_path).resolve())
    return config


def extract_dataset_and_pipeline(config):
    parser = ArgumentParser(add_help=False)
    model_group = ModelParams(parser)
    pipeline_group = PipelineParams(parser)
    dataset = model_group.extract(config)
    pipeline = pipeline_group.extract(config)
    for name in ("convert_SHs_python", "compute_cov3D_python", "debug", "antialiasing"):
        if not hasattr(pipeline, name):
            setattr(pipeline, name, False)
    return dataset, pipeline


def model_ply(model_path, iteration):
    return Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"


def show_interactive_renderer(
    model_path,
    iteration=-1,
    edited_model=None,
    width=512,
    up_axis="z",
):
    """Display sliders that re-render the Gaussian model from a novel camera pose."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as error:
        raise RuntimeError("ipywidgets and IPython are required for the interactive viewer") from error

    model_path = Path(model_path).resolve()
    if iteration < 0:
        values = [
            int(path.name.split("_")[-1])
            for path in (model_path / "point_cloud").glob("iteration_*")
            if path.name.split("_")[-1].isdigit()
        ]
        if not values:
            raise FileNotFoundError(f"No saved iterations under {model_path}")
        iteration = max(values)

    config = read_model_config(model_path)
    dataset, pipeline = extract_dataset_and_pipeline(config)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    available = {"原始模型": gaussians}

    if edited_model:
        edited_path = model_ply(edited_model, iteration)
        if edited_path.is_file():
            edited_gaussians = GaussianModel(dataset.sh_degree)
            edited_gaussians.load_ply(str(edited_path), dataset.train_test_exp)
            available["文本删除后"] = edited_gaussians

    views = scene.getTrainCameras() + scene.getTestCameras()
    reference = (scene.getTestCameras() or scene.getTrainCameras())[0]
    centers = np.stack([view.camera_center.detach().cpu().numpy() for view in views])
    target = np.median(gaussians.get_xyz.detach().cpu().numpy(), axis=0)
    up = axis_vector(up_axis)
    relative = centers - target[None]
    heights = relative @ up
    planar = relative - heights[:, None] * up[None]
    planar_norms = np.linalg.norm(planar, axis=1)
    valid = planar_norms > 1e-5
    if not np.any(valid):
        raise ValueError("Camera centers do not define a usable orbit")
    basis_u = normalized(planar[int(np.flatnonzero(valid)[0])])
    basis_v = normalized(np.cross(up, basis_u))
    radius = float(np.median(planar_norms[valid]))
    default_height = float(np.median(heights))
    default_elevation = math.degrees(math.atan2(default_height, radius))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    model_choice = widgets.ToggleButtons(
        options=list(available),
        description="模型",
        style={"description_width": "85px"},
    )
    azimuth = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=360.0,
        step=2.0,
        description="水平旋转°",
        continuous_update=False,
        style={"description_width": "85px"},
        layout=widgets.Layout(width="560px"),
    )
    elevation = widgets.FloatSlider(
        value=default_elevation,
        min=-35.0,
        max=55.0,
        step=2.0,
        description="俯仰°",
        continuous_update=False,
        style={"description_width": "85px"},
        layout=widgets.Layout(width="560px"),
    )
    zoom = widgets.FloatSlider(
        value=1.0,
        min=0.55,
        max=2.5,
        step=0.05,
        description="缩放",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "85px"},
        layout=widgets.Layout(width="560px"),
    )
    rendered_view = widgets.Image(
        format="png",
        layout=widgets.Layout(
            width=f"{width}px", height=f"{width}px", border="1px solid #ddd"
        ),
    )
    status = widgets.HTML()

    def update(model_name, azimuth_value, elevation_value, zoom_value):
        status.value = "<b>正在重新光栅化...</b>"
        angle = math.radians(float(azimuth_value))
        elevation_radians = math.radians(float(elevation_value))
        orbit_radius = radius / float(zoom_value)
        position = (
            target
            + orbit_radius * (math.cos(angle) * basis_u + math.sin(angle) * basis_v)
            + math.tan(elevation_radians) * orbit_radius * up
        )
        camera = make_camera(reference, position, target, up)
        camera.image_width = width
        camera.image_height = round(width * reference.image_height / reference.image_width)
        with torch.no_grad():
            rendering = render(
                camera,
                available[model_name],
                pipeline,
                background,
                separate_sh=SEPARATE_SH,
            )["render"]
        image = tensor_to_image(rendering)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        rendered_view.value = buffer.getvalue()
        status.value = (
            f"<b>实时 Gaussian rasterizer</b>　{model_name}　"
            f"水平 {azimuth_value:.0f}°　俯仰 {elevation_value:.0f}°　"
            f"缩放 {zoom_value:.2f}×"
        )

    controls_output = widgets.interactive_output(
        update,
        {
            "model_name": model_choice,
            "azimuth_value": azimuth,
            "elevation_value": elevation,
            "zoom_value": zoom,
        },
    )
    title = widgets.HTML(
        "<h3>可操作的 3D Gaussian 渲染模型</h3>"
        "<p>拖动滑块旋转、俯仰和放大缩小。每次操作都会从新的相机位姿重新光栅化；"
        "不是点云，也不是视频。</p>"
    )
    display(
        widgets.VBox(
            [
                title,
                model_choice,
                azimuth,
                elevation,
                zoom,
                status,
                rendered_view,
                controls_output,
            ]
        )
    )
    return {
        "scene": scene,
        "models": available,
        "controls": {
            "model": model_choice,
            "azimuth": azimuth,
            "elevation": elevation,
            "zoom": zoom,
        },
        "image": rendered_view,
    }
