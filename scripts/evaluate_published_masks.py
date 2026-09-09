"""Re-evaluate published Gaussian Grouping masks, without claiming a re-training.

Reads PNGs directly from the authors' ZIP; never extracts arbitrary archive paths.
Only annotation views present in the requested test_mask directory are evaluated.
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from scripts.evaluate_lerf_mask import (
    aggregate_mask_rows, evaluate_prediction_mask, mask_splits,
)


GG_SOURCE = "https://huggingface.co/mqye/Gaussian-Grouping/blob/main/result/lerf_mask.zip"
GG_SHA256 = "537cd7dfd9f17e6b2429d0c992d53c66c97706dda047d36645f0537fe8abc648"


def evaluate(archive_path, test_mask, scene="ramen", expected_sha256=GG_SHA256):
    import numpy as np
    from PIL import Image

    archive_path, test_mask = Path(archive_path), Path(test_mask)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Published prediction archive checksum mismatch")
    splits = mask_splits(test_mask)
    rows, annotation_digest = [], hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive has ambiguous duplicate entries")
        for split in splits:
            for path in sorted(split.glob("*.png")):
                name = f"lerf_mask/{scene}/{split.name}/{path.name}"
                if name not in names:
                    raise ValueError(f"Missing official prediction: {name}")
                if archive.getinfo(name).file_size > 32 * 1024 * 1024:
                    raise ValueError(f"Unexpectedly large mask: {name}")
                prediction = np.asarray(Image.open(io.BytesIO(archive.read(name))).convert("L")) > 128
                target = Image.open(path).convert("L")
                metrics = evaluate_prediction_mask(prediction, None, target, "gg_native", 0.02)
                annotation_digest.update(str(path.relative_to(test_mask)).encode())
                annotation_digest.update(path.read_bytes())
                rows.append({"split": split.name, "label": path.stem, **metrics})
    labels = sorted({row["label"] for row in rows})
    return {
        "model": "Gaussian Grouping (authors' published masks; not retrained here)",
        "dataset": f"LERF-Mask {scene}",
        "source": GG_SOURCE, "prediction_archive_sha256": digest,
        "annotation_sha256": annotation_digest.hexdigest(),
        "protocol": {
            "metric_protocol": "gg_native", "boundary_ratio": 0.02,
            "boundary_pad_edges": True, "resolution": "native_annotation",
            "prediction_resize": "nearest", "aggregation": "macro_class_mean",
            "splits": [split.name for split in splits],
            "supervision_note": "Official masks use tracked object identities and Grounded-SAM matching; not the same supervision as CLIP-only text retrieval.",
            "training_time": "not_measured", "psnr_ssim": "not_available_from_masks",
        },
        **aggregate_mask_rows(rows, labels, "gg_native"), "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_zip", required=True)
    parser.add_argument("--test_mask", required=True)
    parser.add_argument("--scene", default="ramen")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.prediction_zip, args.test_mask, args.scene)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
