"""Plain data types shared by every layer. No UI, TensorFlow or OpenCV here."""
from v7_segmenter.domain.classes import DEFAULT_CLASSES, SegmentClass, classes_from_model
from v7_segmenter.domain.models import DetectedObject, ImageDocument, ModelInfo, Prediction

__all__ = [
    "DEFAULT_CLASSES", "SegmentClass", "classes_from_model",
    "DetectedObject", "ImageDocument", "ModelInfo", "Prediction",
]
