#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

"""Convert input images to the undistorted COLMAP layout used by 3DGS."""

import os
import shutil
import subprocess
from argparse import ArgumentParser


def run(command):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = ArgumentParser("COLMAP converter")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--source_path", "-s", required=True)
    parser.add_argument("--camera", default="OPENCV")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--colmap_executable", default="colmap")
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="magick")
    args = parser.parse_args()

    source = os.path.abspath(args.source_path)
    input_dir = os.path.join(source, "input")
    distorted = os.path.join(source, "distorted")
    database = os.path.join(distorted, "database.db")
    sparse = os.path.join(distorted, "sparse")
    if not os.path.isdir(input_dir):
        parser.error(f"Missing input directory: {input_dir}")
    gpu = "0" if args.no_gpu else "1"

    if not args.skip_matching:
        os.makedirs(sparse, exist_ok=True)
        run([
            args.colmap_executable, "feature_extractor", "--database_path", database,
            "--image_path", input_dir, "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", args.camera,
            "--SiftExtraction.use_gpu", gpu,
        ])
        run([
            args.colmap_executable, f"{args.matcher}_matcher", "--database_path", database,
            "--SiftMatching.use_gpu", gpu,
        ])
        run([
            args.colmap_executable, "mapper", "--database_path", database,
            "--image_path", input_dir, "--output_path", sparse,
            "--Mapper.ba_global_function_tolerance", "0.000001",
        ])

    reconstruction = os.path.join(sparse, "0")
    if not os.path.isdir(reconstruction):
        raise RuntimeError(
            f"No COLMAP reconstruction at {reconstruction}. Capture overlapping views "
            "or use --matcher sequential for ordered video frames."
        )
    run([
        args.colmap_executable, "image_undistorter", "--image_path", input_dir,
        "--input_path", reconstruction, "--output_path", source,
        "--output_type", "COLMAP",
    ])

    sparse_root = os.path.join(source, "sparse")
    sparse_zero = os.path.join(sparse_root, "0")
    os.makedirs(sparse_zero, exist_ok=True)
    for name in os.listdir(sparse_root):
        path = os.path.join(sparse_root, name)
        if name != "0" and os.path.isfile(path):
            shutil.move(path, os.path.join(sparse_zero, name))

    if args.resize:
        image_dir = os.path.join(source, "images")
        for factor in (2, 4, 8):
            destination_dir = os.path.join(source, f"images_{factor}")
            os.makedirs(destination_dir, exist_ok=True)
            for name in os.listdir(image_dir):
                source_file = os.path.join(image_dir, name)
                if not os.path.isfile(source_file):
                    continue
                destination = os.path.join(destination_dir, name)
                shutil.copy2(source_file, destination)
                run([
                    args.magick_executable, "mogrify", "-resize",
                    f"{100 / factor}%", destination,
                ])
    print("COLMAP conversion complete.")


if __name__ == "__main__":
    main()
