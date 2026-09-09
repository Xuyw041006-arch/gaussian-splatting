"""CPU-only GG-native re-scoring of already-thresholded flat LERF masks.

This changes metric geometry/aggregation only. It does not regenerate scores,
change the saved binary predictions, tune thresholds, render, or train a model.
Defaults validate Ramen's three annotated views and six distinct categories.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

from scripts.evaluate_lerf_mask import (
    aggregate_mask_rows, evaluate_prediction_mask, mask_splits,
)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def inspect_masks(predictions, test_mask, expected_views=3, expected_labels=6):
    """Validate all pairs before metric evaluation; never silently skip a mask."""
    from PIL import Image

    predictions, test_mask = Path(predictions).resolve(), Path(test_mask).resolve()
    if not predictions.is_dir():
        raise ValueError(f"Prediction directory is missing: {predictions}")
    if expected_views < 1 or expected_labels < 1:
        raise ValueError("Expected view/category counts must be positive")
    splits = mask_splits(test_mask)
    labels = sorted({path.stem for split in splits for path in split.glob("*.png")})
    if len(splits) != expected_views or len(labels) != expected_labels:
        raise ValueError(f"Expected {expected_views} annotated views and {expected_labels} labels, got {len(splits)} and {len(labels)}")
    rows, annotation_digest, prediction_digest = [], hashlib.sha256(), hashlib.sha256()
    native_views = []
    seen_predictions = set()
    for split in splits:
        prediction_size = target_size = None
        view_rows = []
        for target_path in sorted(split.glob("*.png")):
            prediction_path = predictions / f"{split.name}_{target_path.name}"
            if not prediction_path.is_file():
                raise ValueError(f"Missing saved prediction: {prediction_path.name}")
            if target_path.is_symlink() or prediction_path.is_symlink():
                raise ValueError("Mask inputs must be regular files, not symlinks")
            if prediction_path.name in seen_predictions:
                raise ValueError("Flat mask names ambiguously identify more than one annotation")
            seen_predictions.add(prediction_path.name)
            target_bytes, prediction_bytes = target_path.read_bytes(), prediction_path.read_bytes()
            try:
                with Image.open(io.BytesIO(target_bytes)) as target, Image.open(io.BytesIO(prediction_bytes)) as prediction:
                    target.load()
                    prediction.load()
                    if prediction.mode not in ("1", "L"):
                        raise ValueError("Saved predictions must be binary grayscale masks, not colored overlays")
                    colors = prediction.convert("L").getcolors(maxcolors=3)
                    if colors is None or not {value for _, value in colors}.issubset({0, 255}):
                        raise ValueError("Prediction contains nonbinary values; this tool cannot re-threshold score maps")
                    p_size, t_size = prediction.size, target.size
            except (OSError, SyntaxError) as error:
                raise ValueError(f"Unreadable PNG mask: {prediction_path.name} / {target_path.name}") from error
            if min(*p_size, *t_size) <= 0:
                raise ValueError("Mask dimensions must be positive")
            if ((prediction_size is not None and prediction_size != p_size)
                    or (target_size is not None and target_size != t_size)):
                raise ValueError(f"Inconsistent mask dimensions within annotated view {split.name}")
            # Normal integer resize rounding is allowed; a rotated/cropped or
            # unrelated prediction grid is not silently stretched into the GT.
            if abs(p_size[0] * t_size[1] - p_size[1] * t_size[0]) > max(*p_size, *t_size):
                raise ValueError(f"Prediction/GT aspect ratios disagree for {prediction_path.name}")
            prediction_size, target_size = p_size, t_size
            relative = str(target_path.relative_to(test_mask))
            # Deliberately identical to evaluate_published_masks.evaluate:
            # sorted split/file order; relative path bytes then original PNG.
            annotation_digest.update(relative.encode())
            annotation_digest.update(target_bytes)
            prediction_digest.update(prediction_path.name.encode())
            prediction_digest.update(prediction_bytes)
            record = {
                "split": split.name, "label": target_path.stem,
                "prediction_path": str(prediction_path), "annotation_path": str(target_path),
                "prediction_sha256": digest_bytes(prediction_bytes), "annotation_sha256": digest_bytes(target_bytes),
                "prediction_width": p_size[0], "prediction_height": p_size[1],
                "native_width": t_size[0], "native_height": t_size[1],
            }
            rows.append(record)
            view_rows.append({"label": target_path.stem, "relative_path": relative,
                              "sha256": record["annotation_sha256"], "width": t_size[0], "height": t_size[1]})
        native_views.append({"split": split.name, "labels": view_rows})
    fingerprint = digest_bytes(json.dumps(native_views, sort_keys=True, separators=(",", ":")).encode())
    return {
        "predictions": str(predictions), "test_mask": str(test_mask), "rows": rows,
        "splits": [split.name for split in splits], "labels": labels,
        "annotation_sha256": annotation_digest.hexdigest(),
        "prediction_masks_sha256": prediction_digest.hexdigest(),
        "native_annotation_fingerprint": fingerprint,
        "native_annotation_views": native_views,
    }


def evaluate(predictions, test_mask, expected_views=3, expected_labels=6):
    import numpy as np
    from PIL import Image

    inspected = inspect_masks(predictions, test_mask, expected_views, expected_labels)
    rows = []
    for pair in inspected["rows"]:
        # Guard against concurrent notebook cells replacing a mask after audit.
        prediction_bytes, target_bytes = Path(pair["prediction_path"]).read_bytes(), Path(pair["annotation_path"]).read_bytes()
        if (digest_bytes(prediction_bytes) != pair["prediction_sha256"]
                or digest_bytes(target_bytes) != pair["annotation_sha256"]):
            raise ValueError("A prediction or annotation changed during re-scoring")
        with Image.open(io.BytesIO(prediction_bytes)) as prediction, Image.open(io.BytesIO(target_bytes)) as target:
            metrics = evaluate_prediction_mask(np.asarray(prediction.convert("L")) > 128,
                                               None, target.convert("L"), "gg_native", 0.02)
        rows.append({**pair, **metrics})
    metrics_path = Path(predictions) / "metrics.json"
    source_metrics = None
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text())
        source_metrics = {"path": str(metrics_path.resolve()), "sha256": digest_bytes(metrics_path.read_bytes()),
                          **{key: payload.get(key) for key in ("model", "iteration", "threshold", "granularity", "protocol", "weight_selection")}}
    return {
        "operation": "rescore_saved_binary_predictions_only",
        "model": "Saved binary masks from local evaluator; no new inference or training",
        "dataset": "LERF-Mask annotated subset",
        "prediction_directory": inspected["predictions"], "annotation_directory": inspected["test_mask"],
        "source_metrics": source_metrics,
        "annotation_sha256": inspected["annotation_sha256"],
        "prediction_masks_sha256": inspected["prediction_masks_sha256"],
        "native_annotation_fingerprint": inspected["native_annotation_fingerprint"],
        "native_annotation_views": inspected["native_annotation_views"],
        "rescorer_sha256": digest_bytes(Path(__file__).read_bytes()),
        "protocol": {
            "metric_protocol": "gg_native", "boundary_ratio": 0.02,
            "boundary_pad_edges": True, "resolution": "native_annotation",
            "prediction_resize": "nearest", "aggregation": "macro_class_mean",
            "splits": inspected["splits"], "labels": inspected["labels"],
            "source_binary_predictions_unchanged": True,
            "score_regeneration": False, "threshold_tuning": False,
            "binary_foreground": "uint8 > 128 (input already restricted to 0 or 255)",
            "annotation_digest_protocol": "evaluate_published_masks: sorted relative_path.encode() + original PNG bytes",
            "training_time": "not_measured", "psnr_ssim": "not_available_from_masks",
            "interpretation": "Only the mask metric protocol changed; this cannot establish a model or retrieval improvement.",
        },
        **aggregate_mask_rows(rows, inspected["labels"], "gg_native"), "rows": rows,
    }


def write_result(output, result):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test_mask", required=True)
    parser.add_argument("--output", required=True, help="New JSON only; existing output is never overwritten")
    parser.add_argument("--expected_views", type=int, default=3)
    parser.add_argument("--expected_labels", type=int, default=6)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new JSON path")
    try:
        result = evaluate(args.predictions, args.test_mask, args.expected_views, args.expected_labels)
        write_result(args.output, result)
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "native_annotation_views")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
