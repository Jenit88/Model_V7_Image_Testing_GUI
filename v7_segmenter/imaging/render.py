"""Drawing predicted objects over a photo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from v7_segmenter.domain import DetectedObject, SegmentClass

HIGHLIGHT_RGB = (255, 255, 255)


@dataclass(frozen=True)
class OverlayStyle:
    opacity: float = 0.45       # 0 = outline only, 1 = solid colour
    outlines: bool = True
    outline_px: int = 2


def class_lookup(objects: Iterable[DetectedObject], hidden_classes: set[int],
                 min_confidence: float, size: int) -> np.ndarray:
    """Class id per object id; 0 for background and for filtered-out objects."""
    lookup = np.zeros(max(1, size), dtype=np.uint8)
    for item in objects:
        if (0 < item.id < len(lookup) and item.class_id not in hidden_classes
                and item.confidence >= min_confidence):
            lookup[item.id] = item.class_id
    return lookup


def colour_table(classes: Iterable[SegmentClass]) -> np.ndarray:
    """RGB per class id (row 0 = background)."""
    classes = tuple(classes)
    table = np.zeros((max((c.id for c in classes), default=0) + 1, 3), dtype=np.uint8)
    for item in classes:
        table[item.id] = item.rgb
    return table


def resize_nearest(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize of a label map, for any integer dtype."""
    source_h, source_w = labels.shape[:2]
    rows = np.minimum(((np.arange(height) + 0.5) * source_h / height).astype(np.int64), source_h - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * source_w / width).astype(np.int64), source_w - 1)
    return labels[rows[:, None], cols[None, :]]


def _edges(instances: np.ndarray) -> np.ndarray:
    """Pixels on either side of a change of object id."""
    edge = np.zeros(instances.shape, dtype=bool)
    change = instances[:, 1:] != instances[:, :-1]
    edge[:, 1:] |= change
    edge[:, :-1] |= change
    change = instances[1:, :] != instances[:-1, :]
    edge[1:, :] |= change
    edge[:-1, :] |= change
    return edge


def _thicken(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 1:
        return mask
    return cv2.dilate(mask.astype(np.uint8), np.ones((px, px), np.uint8)).astype(bool)


def compose_overlay(rgb: np.ndarray, instances: np.ndarray, lookup: np.ndarray,
                    colours: np.ndarray, style: OverlayStyle,
                    highlight_id: int | None = None) -> np.ndarray:
    """Tint and outline every object whose lookup entry is non-zero.

    `rgb` and `instances` must have the same height and width; they are usually
    the visible part of the image already scaled to screen size, so outlines
    stay crisp at any zoom.
    """
    classes = lookup[np.minimum(instances, len(lookup) - 1)]
    shown = classes > 0
    out = rgb.copy()
    if shown.any():
        alpha = float(style.opacity)
        if alpha > 0:
            tint = colours[classes[shown]].astype(np.float32)
            out[shown] = (rgb[shown].astype(np.float32) * (1 - alpha) + tint * alpha
                          ).astype(np.uint8)
        if style.outlines or alpha == 0:
            edge = _thicken(_edges(instances), style.outline_px) & shown
            out[edge] = colours[classes[edge]]
    if highlight_id is not None and highlight_id > 0:
        target = instances == highlight_id
        if target.any():
            ring = _thicken(_edges(target.astype(np.uint8)), style.outline_px + 2)
            out[ring] = HIGHLIGHT_RGB
    return out


def full_resolution_overlay(rgb: np.ndarray, instances: np.ndarray, lookup: np.ndarray,
                            colours: np.ndarray, style: OverlayStyle) -> np.ndarray:
    """The overlay at the photo's own size, for saving; outlines scale with it."""
    px = max(style.outline_px, int(round(max(rgb.shape[:2]) / 700)))
    return compose_overlay(rgb, instances, lookup, colours,
                           OverlayStyle(style.opacity, style.outlines, px))
