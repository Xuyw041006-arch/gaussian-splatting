"""Create a safe custom-scene input folder from user images."""

import shutil
from argparse import ArgumentParser
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    parser = ArgumentParser(description="Prepare custom images for COLMAP")
    parser.add_argument("--images", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.images).resolve()
    destination = Path(args.scene).resolve() / "input"
    if not source.is_dir():
        parser.error(f"Image directory does not exist: {source}")
    images = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if len(images) < 3:
        parser.error("At least three images are required; 8-30 overlapping views are recommended")
    destination.mkdir(parents=True, exist_ok=True)
    existing = [destination / path.name for path in images if (destination / path.name).exists()]
    if existing and not args.overwrite:
        parser.error(f"{len(existing)} destination images already exist; pass --overwrite")
    for path in images:
        shutil.copy2(path, destination / path.name)
    print(f"Prepared {len(images)} images in {destination}")


if __name__ == "__main__":
    main()
