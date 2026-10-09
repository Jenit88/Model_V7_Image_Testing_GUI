"""Image documents, predictions and model information."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ImageDocument:
    """A photo opened in the application.

    `raw` is the array exactly as read from disk; `display_rgb` is the uint8 RGB
    image shown on screen. After a prediction the latter is replaced by the
    model's own conversion of `raw`, so the overlay sits on what the model saw.
    """

    path: Path
    raw: np.ndarray
    display_rgb: np.ndarray

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def width(self) -> int:
        return int(self.display_rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.display_rgb.shape[0])


@dataclass(frozen=True)
class DetectedObject:
    id: int
    class_id: int
    class_name: str
    confidence: float
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    touches_border: bool = False

    @property
    def centre_text(self) -> str:
        return f"{self.centroid_xy[0]:.0f}, {self.centroid_xy[1]:.0f}"


@dataclass
class Prediction:
    """One prediction for one photo, at the photo's own resolution."""

    image_path: Path
    instance_map: np.ndarray            # (H, W) object id per pixel, 0 = background
    objects: list[DetectedObject]
    model_ms: float
    decode_ms: float
    output_dir: Path                    # model_v7's own artefacts for this prediction
    artifacts: dict[str, Path] = field(default_factory=dict)
    below_confidence_floor: int = 0
    wall_seconds: float = 0.0           # everything, including saving the artefacts

    def object_by_id(self, object_id: int) -> DetectedObject | None:
        for item in self.objects:
            if item.id == object_id:
                return item
        return None

    def counts_by_class(self) -> Counter:
        return Counter(item.class_id for item in self.objects)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    file: Path
    sha256_prefix: str
    device: str
    load_seconds: float
    card: dict = field(default_factory=dict)  # provenance and scores from config/model_card.json
