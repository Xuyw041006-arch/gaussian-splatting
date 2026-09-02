"""Small browser UI for text search and click inspection on registered views.

Use the official SIBR viewer for free 3D navigation; this companion UI focuses
on semantics and exact click-to-Gaussian information.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from plyfile import PlyData

from semantic.artifact import cosine_scores, decode_features, project_clip_feature
from semantic.inspection import pick_point, project_points
from semantic_query import latest_iteration


class SemanticViewer:
    def __init__(self, model_path, source_path, iteration, device):
        self.model_path = Path(model_path).resolve()
        self.source_path = Path(source_path).resolve()
        self.iteration = latest_iteration(self.model_path) if iteration < 0 else iteration
        with open(self.model_path / "cameras.json", encoding="utf-8") as handle:
            self.cameras = json.load(handle)
        self.camera_by_name = {camera["img_name"]: camera for camera in self.cameras}

        ply = PlyData.read(
            self.model_path / "point_cloud" / f"iteration_{self.iteration}" / "point_cloud.ply"
        )
        self.vertices = ply["vertex"].data
        self.xyz = np.column_stack([self.vertices[axis] for axis in ("x", "y", "z")])
        artifact = torch.load(
            self.model_path / "semantic" / f"iteration_{self.iteration}" / "semantic_features.pt",
            map_location="cpu",
        )
        self.semantic_features = decode_features(
            artifact["features"].float().numpy(),
            artifact["feature_min"].numpy(), artifact["feature_max"].numpy(),
        )
        self.pca_mean = artifact["pca_mean"].numpy()
        self.pca_components = artifact["pca_components"].numpy()

        try:
            import open_clip
        except ImportError as error:
            raise RuntimeError(f"Install requirements-semantic.txt: {error}") from error
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            artifact["clip_model"], pretrained=artifact["clip_pretrained"],
            precision="fp16" if self.device.type == "cuda" else "fp32",
        )
        self.clip_model = self.clip_model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(artifact["clip_model"])

    def image(self, name):
        path = self.source_path / "images" / name
        if not path.is_file():
            path = self.source_path / "images" / Path(name).name
        return Image.open(path).convert("RGB")

    def scores(self, prompt):
        if not prompt.strip():
            return np.zeros(len(self.xyz), dtype=np.float32)
        with torch.no_grad():
            feature = torch.nn.functional.normalize(
                self.clip_model.encode_text(self.tokenizer([prompt]).to(self.device)).float(),
                dim=-1, p=2,
            )[0].cpu().numpy()
        query = project_clip_feature(feature, self.pca_mean, self.pca_components)
        return cosine_scores(self.semantic_features, query)

    def overlay(self, name, prompt, threshold):
        image = self.image(name)
        camera = self.camera_by_name[name]
        scores = self.scores(prompt)
        chosen = np.flatnonzero(scores >= threshold)
        if len(chosen) > 30000:
            chosen = chosen[np.argsort(scores[chosen])[-30000:]]
        pixels, depth = project_points(self.xyz[chosen], camera)
        valid = (
            (depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera["width"])
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera["height"])
        )
        draw = ImageDraw.Draw(image, "RGBA")
        for x, y in pixels[valid]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 40, 40, 150))
        status = {
            "query": prompt, "threshold": threshold,
            "matched_gaussians": int((scores >= threshold).sum()),
            "visible_overlay_points": int(valid.sum()),
        }
        return image, status

    def inspect(self, name, prompt, threshold, x, y):
        camera = self.camera_by_name[name]
        scores = self.scores(prompt)
        candidate_mask = scores >= threshold if prompt.strip() else None
        index = pick_point(self.xyz, camera, x, y, radius=10, candidate_mask=candidate_mask)
        if index is None and candidate_mask is not None:
            index = pick_point(self.xyz, camera, x, y, radius=10)
        if index is None:
            return {"pixel": [x, y], "found": False}
        return {
            "pixel": [x, y], "found": True, "gaussian_index": index,
            "xyz": self.xyz[index].tolist(),
            "opacity": float(1.0 / (1.0 + np.exp(-self.vertices["opacity"][index]))),
            "query": prompt or None,
            "query_score": float(scores[index]) if prompt.strip() else None,
        }


def main():
    parser = ArgumentParser(description="Semantic search and click inspection UI")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    try:
        import gradio as gr
    except ImportError as error:
        parser.error(f"Install requirements-ui.txt: {error}")

    viewer = SemanticViewer(args.model, args.source, args.iteration, args.device)
    names = list(viewer.camera_by_name)
    with gr.Blocks(title="Semantic Adaptive 3DGS") as demo:
        gr.Markdown("# Semantic Adaptive 3DGS\n输入文本查找物体；点击图像查看对应高斯信息。")
        with gr.Row():
            view = gr.Dropdown(names, value=names[0], label="Registered view")
            prompt = gr.Textbox(value="apple", label="Semantic query")
            threshold = gr.Slider(-1, 1, value=0.25, step=0.01, label="Cosine threshold")
        search = gr.Button("Search all matching Gaussians", variant="primary")
        image = gr.Image(value=viewer.image(names[0]), type="pil", label="Click an object")
        info = gr.JSON(label="Result / clicked Gaussian")

        view.change(lambda name: viewer.image(name), inputs=view, outputs=image)
        search.click(viewer.overlay, inputs=[view, prompt, threshold], outputs=[image, info])

        def on_select(name, text, score_threshold, event: gr.SelectData):
            x, y = event.index
            return viewer.inspect(name, text, score_threshold, x, y)

        image.select(on_select, inputs=[view, prompt, threshold], outputs=info)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
