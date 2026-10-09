"""The application state: a single observable object that the views render.

Only the controller changes it; every change publishes a topic on the event
bus, and each view re-reads what it needs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from v7_segmenter.domain import (
    DEFAULT_CLASSES, DetectedObject, ImageDocument, ModelInfo, Prediction, SegmentClass,
)
from v7_segmenter.events import EventBus


class Topic:
    MODEL = "model"             # model status, information and classes
    DOCUMENT = "document"       # a photo was opened or cleared
    PREDICTION = "prediction"   # a prediction was made or cleared
    DISPLAY = "display"         # overlay options changed
    SELECTION = "selection"     # the selected object changed
    ACTIVITY = "activity"       # busy state or status message changed


class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


# (key, label, model_v7 artefact) -- the last two layers onward exist only once
# a prediction has been made.
LAYERS: tuple[tuple[str, str, str | None], ...] = (
    ("overlay", "Segments on photo", None),
    ("original", "Photo only", None),
    ("model_overlay", "Model overlay (boxes and labels)", "instance_overlay"),
    ("semantic", "Semantic classes", "semantic_colour"),
    ("boundary", "Boundary probability", "boundary_heatmap"),
    ("distance", "Inner distance", "inner_distance"),
    ("confidence", "Semantic confidence", "semantic_confidence"),
)


@dataclass(frozen=True)
class DisplayOptions:
    show_segments: bool = True
    opacity: float = 0.45
    outlines: bool = True
    min_confidence: float = 0.0       # a view filter only; the model is unchanged
    hidden_classes: frozenset = frozenset()
    layer: str = "overlay"


class AppState:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.model_status = ModelStatus.LOADING
        self.model_info: ModelInfo | None = None
        self.model_error: str | None = None
        self.classes: tuple[SegmentClass, ...] = DEFAULT_CLASSES
        self.document: ImageDocument | None = None
        self.prediction: Prediction | None = None
        self.display = DisplayOptions()
        self.selected_id: int | None = None
        self.predicting = False
        self.message = ""
        self.folder_position: tuple[int, int] | None = None   # (1-based index, total)

    # ---- model -------------------------------------------------------------
    def model_ready(self, info: ModelInfo, classes: tuple[SegmentClass, ...]) -> None:
        self.model_status, self.model_info, self.classes = ModelStatus.READY, info, classes
        self.model_error = None
        self.bus.publish(Topic.MODEL)

    def model_failed(self, error: str) -> None:
        self.model_status, self.model_error = ModelStatus.FAILED, error
        self.bus.publish(Topic.MODEL)

    # ---- photo and prediction ---------------------------------------------
    def set_document(self, document: ImageDocument | None,
                     folder_position: tuple[int, int] | None = None) -> None:
        self.document = document
        self.folder_position = folder_position if document is not None else None
        self.prediction = None
        self.selected_id = None
        if self.display.layer not in ("overlay", "original"):
            self.display = replace(self.display, layer="overlay")
        for topic in (Topic.DOCUMENT, Topic.PREDICTION, Topic.SELECTION, Topic.DISPLAY):
            self.bus.publish(topic)

    def set_prediction(self, prediction: Prediction | None) -> None:
        self.prediction = prediction
        self.selected_id = None
        self.bus.publish(Topic.PREDICTION)
        self.bus.publish(Topic.SELECTION)

    # ---- view options and selection ---------------------------------------
    def update_display(self, **changes) -> None:
        self.display = replace(self.display, **changes)
        selected = self.selected_object
        if selected is not None and not self.is_visible(selected):
            self.selected_id = None
            self.bus.publish(Topic.SELECTION)
        self.bus.publish(Topic.DISPLAY)

    def select(self, object_id: int | None) -> None:
        if object_id != self.selected_id:
            self.selected_id = object_id
            self.bus.publish(Topic.SELECTION)

    def set_activity(self, *, predicting: bool | None = None, message: str | None = None) -> None:
        if predicting is not None:
            self.predicting = predicting
        if message is not None:
            self.message = message
        self.bus.publish(Topic.ACTIVITY)

    # ---- queries -------------------------------------------------------------
    @property
    def is_model_ready(self) -> bool:
        return self.model_status is ModelStatus.READY

    @property
    def selected_object(self) -> DetectedObject | None:
        if self.prediction is None or self.selected_id is None:
            return None
        return self.prediction.object_by_id(self.selected_id)

    def is_visible(self, item: DetectedObject) -> bool:
        return (item.class_id not in self.display.hidden_classes
                and item.confidence >= self.display.min_confidence)

    def visible_objects(self) -> list[DetectedObject]:
        if self.prediction is None:
            return []
        return [item for item in self.prediction.objects if self.is_visible(item)]

    def class_by_id(self, class_id: int) -> SegmentClass | None:
        return next((c for c in self.classes if c.id == class_id), None)
