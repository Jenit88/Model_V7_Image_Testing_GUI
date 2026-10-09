"""Saving results: an overlay image, or a folder with every output."""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from v7_segmenter.domain import ImageDocument, ModelInfo, Prediction, SegmentClass
from v7_segmenter.imaging.io import write_image
from v7_segmenter.imaging.render import (
    OverlayStyle, class_lookup, colour_table, full_resolution_overlay,
)
from v7_segmenter.services.state import DisplayOptions

# Written by model_v7.predict_one_image for every prediction.
MODEL_OUTPUT_FILES = (
    "instances.json", "instances.png", "semantic_colour.png", "boundary.png",
    "inner_distance.png", "semantic_confidence.png", "instance_ids.npy", "semantic_ids.npy",
)


def render_overlay(document: ImageDocument, prediction: Prediction, display: DisplayOptions,
                   classes: tuple[SegmentClass, ...]) -> np.ndarray:
    """The overlay at full resolution, with the classes and filter shown on screen."""
    lookup = class_lookup(prediction.objects, set(display.hidden_classes),
                          display.min_confidence, int(prediction.instance_map.max()) + 1)
    return full_resolution_overlay(document.display_rgb, prediction.instance_map, lookup,
                                   colour_table(classes),
                                   OverlayStyle(display.opacity, display.outlines))


def save_overlay(path: Path, document: ImageDocument, prediction: Prediction,
                 display: DisplayOptions, classes: tuple[SegmentClass, ...]) -> Path:
    write_image(path, render_overlay(document, prediction, display, classes))
    return path


def _unique_folder(base: Path) -> Path:
    candidate, number = base, 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{number}")
        number += 1
    return candidate


def export_results(folder: Path, document: ImageDocument, prediction: Prediction,
                   display: DisplayOptions, classes: tuple[SegmentClass, ...],
                   model: ModelInfo | None) -> Path:
    """Create <folder>/<photo>_v7_results/ with every output for this photo."""
    target = _unique_folder(folder / f"{document.path.stem}_v7_results")
    target.mkdir(parents=True)
    for name in MODEL_OUTPUT_FILES:
        source = prediction.output_dir / name
        if source.is_file():
            shutil.copy2(source, target / name)
    write_image(target / "overlay.png", render_overlay(document, prediction, display, classes))

    names = {c.id: c.name for c in classes}
    with (target / "objects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "class", "confidence", "area_px", "bbox_x0", "bbox_y0",
                         "bbox_x1", "bbox_y1", "centroid_x", "centroid_y",
                         "touches_border", "shown_in_overlay"])
        for item in prediction.objects:
            shown = (item.class_id not in display.hidden_classes
                     and item.confidence >= display.min_confidence)
            writer.writerow([item.id, names.get(item.class_id, item.class_name),
                             f"{item.confidence:.4f}", item.area_px, *item.bbox_xyxy,
                             f"{item.centroid_xy[0]:.1f}", f"{item.centroid_xy[1]:.1f}",
                             int(item.touches_border), int(shown)])

    counts = prediction.counts_by_class()
    summary = {
        "photo": str(document.path),
        "size": [document.width, document.height],
        "exported": datetime.now().isoformat(timespec="seconds"),
        "model": None if model is None else {
            "name": model.name, "file": model.file.name, "sha256_prefix": model.sha256_prefix,
            "device": model.device,
        },
        "objects": len(prediction.objects),
        "objects_per_class": {names.get(cid, str(cid)): counts.get(cid, 0) for cid in names},
        "overlay_options": {
            "opacity": display.opacity, "outlines": display.outlines,
            "min_confidence": display.min_confidence,
            "hidden_classes": sorted(names.get(c, str(c)) for c in display.hidden_classes),
        },
        "timing_ms": {"model": prediction.model_ms, "decoding": prediction.decode_ms},
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return target
