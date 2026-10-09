"""Adapter around model_v7.py, the only module that touches TensorFlow.

model_v7.py is used read-only: it is imported as a module (its custom layers
register on import, which the saved model needs) and every prediction is made
by its own predict_one_image -- letterbox to 512, model, instance decoding,
confidence floor, and restoration to the photo's own size. This adapter only
converts that function's result into domain objects.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from v7_segmenter.domain import (
    DEFAULT_CLASSES, DetectedObject, ModelInfo, Prediction, SegmentClass, classes_from_model,
)
from v7_segmenter.paths import AppPaths

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def is_lfs_pointer(path: Path) -> bool:
    """True if `path` is the small text stub Git leaves when Git LFS was not
    installed at clone time, rather than the model itself."""
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as handle:
            return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def sha256_prefix(path: Path, length: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


class V7Engine:
    MODULE_NAME = "model_v7"

    def __init__(self, paths: AppPaths):
        self.paths = paths
        self._code = None
        self._model = None
        self._classes: tuple[SegmentClass, ...] = DEFAULT_CLASSES
        self.info: ModelInfo | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def classes(self) -> tuple[SegmentClass, ...]:
        return self._classes

    # ---- loading -----------------------------------------------------------
    def load(self) -> ModelInfo:
        started = time.perf_counter()
        for required in (self.paths.model_code, self.paths.model_file):
            if not required.is_file():
                raise ModelLoadError(f"Missing file: {required}")
        if is_lfs_pointer(self.paths.model_file):
            raise ModelLoadError(
                f"{self.paths.model_file.name} is only a Git LFS placeholder, not the model. "
                "Install Git LFS (included with Git for Windows), then run "
                "'git lfs install' and 'git lfs pull' in the repository folder.")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        code = self._import_model_code()
        import tensorflow as tf

        model = tf.keras.models.load_model(self.paths.model_file, compile=False)
        code.validate_model_output_shapes(model)
        # One throw-away pass, so the first real prediction does not also pay
        # for TensorFlow tracing the graph.
        blank = np.full((1, code.IMG_SIZE, code.IMG_SIZE, 3),
                        code.LETTERBOX_FILL_VALUE / 255.0, dtype=np.float32)
        code.unpack_model_outputs(model(blank, training=False))

        card = self._read_model_card()
        digest = sha256_prefix(self.paths.model_file)
        expected = str(card.get("sha256_prefix", ""))
        if expected and not digest.startswith(expected[:12]):
            log.warning("model file sha256 %s does not match model_card.json (%s)", digest, expected)
        self._code, self._model = code, model
        self._classes = classes_from_model(code.CLASS_NAMES, code.CLASS_COLOURS_RGB)
        self.info = ModelInfo(
            name=str(card.get("name", self.paths.model_file.stem)),
            file=self.paths.model_file,
            sha256_prefix=digest,
            device="GPU" if tf.config.list_physical_devices("GPU") else "CPU",
            load_seconds=time.perf_counter() - started,
            card=card,
        )
        log.info("model ready in %.1f s on %s (sha256 %s)", self.info.load_seconds,
                 self.info.device, digest)
        return self.info

    def _import_model_code(self):
        loaded = sys.modules.get(self.MODULE_NAME)
        if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == \
                self.paths.model_code.resolve():
            return loaded
        spec = importlib.util.spec_from_file_location(self.MODULE_NAME, self.paths.model_code)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.MODULE_NAME] = module
        # Leave model_v7_resource/ exactly as it is: no __pycache__ next to model_v7.py.
        writes_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = writes_bytecode
        return module

    def _read_model_card(self) -> dict:
        try:
            return json.loads(self.paths.model_card.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("no readable model card at %s", self.paths.model_card)
            return {}

    # ---- prediction ----------------------------------------------------------
    def model_rgb(self, raw: np.ndarray) -> np.ndarray:
        """model_v7's own uint8 RGB conversion of a photo, i.e. what it predicts on."""
        return self._code._as_rgb_uint8(raw, name="photo")

    def predict(self, image_path: Path, output_dir: Path) -> Prediction:
        if not self.ready:
            raise RuntimeError("The model has not finished loading.")
        started = time.perf_counter()
        result = self._code.predict_one_image(self._model, Path(image_path), Path(output_dir),
                                              model_path=self.paths.model_file)
        artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
        instance_map = np.load(artifacts["instance_ids"])
        objects = [
            DetectedObject(
                id=int(item["instance_id"]),
                class_id=int(item["class_id"]),
                class_name=str(item["class_name"]),
                confidence=float(item["confidence"]),
                area_px=int(item["area_pixels"]),
                bbox_xyxy=tuple(int(v) for v in item["bbox_xyxy"]),
                centroid_xy=(float(item["centroid_xy"][0]), float(item["centroid_xy"][1])),
                touches_border=bool(item.get("touches_image_border", False)),
            )
            for item in result["instances"]
        ]
        timing = result.get("timing_ms", {})
        prediction = Prediction(
            image_path=Path(image_path),
            instance_map=instance_map,
            objects=objects,
            model_ms=float(timing.get("model_inference_and_device_transfer", 0.0)),
            decode_ms=float(timing.get("probability_conversion_and_instance_decoding", 0.0)),
            output_dir=Path(output_dir),
            artifacts=artifacts,
            below_confidence_floor=int(result.get("instances_below_confidence_floor", 0)),
            wall_seconds=time.perf_counter() - started,
        )
        log.info("predicted %d objects for %s in %.2f s", len(objects), image_path,
                 prediction.wall_seconds)
        return prediction
