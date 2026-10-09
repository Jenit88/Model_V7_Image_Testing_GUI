"""The central canvas: the insert-image placeholder, or the photo with its
prediction, with zoom (mouse wheel), pan (drag) and object picking (click)."""
from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from v7_segmenter.domain import DetectedObject
from v7_segmenter.imaging.io import read_image, to_display_uint8
from v7_segmenter.imaging.render import (
    OverlayStyle, class_lookup, colour_table, compose_overlay, resize_nearest,
)
from v7_segmenter.imaging.viewport import Viewport
from v7_segmenter.services.state import LAYERS, AppState, Topic
from v7_segmenter.ui import theme
from v7_segmenter.ui.theme import px

log = logging.getLogger(__name__)
ARTIFACT_OF_LAYER = {key: artifact for key, _, artifact in LAYERS}
DRAG_THRESHOLD = 4


class ImageView(ttk.Frame):
    def __init__(self, master: tk.Misc, state: AppState, scale: float):
        super().__init__(master)
        self.state = state
        self.scale = scale
        self.viewport = Viewport()
        self.canvas = tk.Canvas(self, bg=theme.CANVAS_BG, highlightthickness=0, takefocus=1)
        self.canvas.pack(fill="both", expand=True)

        # Set by the controller.
        self.on_insert: Callable[[], None] = lambda: None
        self.on_select: Callable[[int | None], None] = lambda object_id: None
        self.on_hover: Callable[[str], None] = lambda text: None
        self.on_zoom: Callable[[float | None], None] = lambda zoom: None

        self._photo = None
        self._render_job = None
        self._hover = False
        self._press = None
        self._last = None
        self._dragging = False
        self._layer_images: dict[str, np.ndarray] = {}
        self._lookup_key = None
        self._lookup = np.zeros(1, dtype=np.uint8)

        canvas = self.canvas
        canvas.bind("<Configure>", self._on_configure)
        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        canvas.bind("<MouseWheel>", self._on_wheel)
        canvas.bind("<Motion>", self._on_motion)
        canvas.bind("<Leave>", self._on_leave)

        bus = state.bus
        bus.subscribe(Topic.DOCUMENT, self._on_document)
        bus.subscribe(Topic.PREDICTION, self._on_prediction)
        for topic in (Topic.DISPLAY, Topic.SELECTION, Topic.ACTIVITY, Topic.MODEL):
            bus.subscribe(topic, self.refresh)

    # ---- commands ------------------------------------------------------------
    def refresh(self) -> None:
        if self._render_job is None:
            self._render_job = self.after(15, self._render)

    def zoom_in(self) -> None:
        self._zoom(1.25)

    def zoom_out(self) -> None:
        self._zoom(0.8)

    def fit(self) -> None:
        if self.viewport.has_image:
            self.viewport.fit()
            self._zoom_changed()

    def actual_size(self) -> None:
        if self.viewport.has_image:
            self.viewport.set_zoom(1.0)
            self._zoom_changed()

    def ensure_visible(self, item: DetectedObject) -> None:
        if not self.viewport.contains_box(*item.bbox_xyxy):
            self.viewport.center_on(*item.centroid_xy)
            self.refresh()

    # ---- state changes -------------------------------------------------------
    def _on_document(self) -> None:
        document = self.state.document
        self._layer_images.clear()
        if document is None:
            self.viewport.set_image(0, 0)
            self.on_zoom(None)
        else:
            self.viewport.set_image(document.width, document.height)
            self.on_zoom(self.viewport.zoom)
        self.refresh()

    def _on_prediction(self) -> None:
        self._layer_images.clear()
        self._lookup_key = None
        self.refresh()

    # ---- mouse -----------------------------------------------------------------
    def _zoom(self, factor: float, anchor=None) -> None:
        if self.viewport.has_image:
            self.viewport.zoom_by(factor, anchor)
            self._zoom_changed()

    def _zoom_changed(self) -> None:
        self.on_zoom(self.viewport.zoom)
        self.refresh()

    def _on_configure(self, event) -> None:
        self.viewport.set_view(event.width, event.height)
        if self.viewport.has_image:
            self.on_zoom(self.viewport.zoom)
        self.refresh()

    def _on_press(self, event) -> None:
        self.canvas.focus_set()
        self._press = self._last = (event.x, event.y)
        self._dragging = False

    def _on_drag(self, event) -> None:
        if self._press is None or self.state.document is None:
            return
        if not self._dragging:
            moved = abs(event.x - self._press[0]) + abs(event.y - self._press[1])
            if moved < px(DRAG_THRESHOLD, self.scale):
                return
            self._dragging = True
            self.canvas.configure(cursor="fleur")
        self.viewport.pan(event.x - self._last[0], event.y - self._last[1])
        self._last = (event.x, event.y)
        self.refresh()

    def _on_release(self, event) -> None:
        was_drag = self._dragging
        self._press = self._last = None
        self._dragging = False
        self._update_cursor()
        if was_drag:
            return
        if self.state.document is None:
            self.on_insert()
            return
        hit = self._object_at(event.x, event.y)
        self.on_select(hit.id if hit is not None else None)

    def _on_wheel(self, event) -> None:
        self._zoom(1.25 if event.delta > 0 else 0.8, (event.x, event.y))

    def _on_motion(self, event) -> None:
        if self.state.document is None:
            self._set_placeholder_hover(True)
            return
        x, y = self.viewport.screen_to_image(event.x, event.y)
        document = self.state.document
        if not (0 <= x < document.width and 0 <= y < document.height):
            self.on_hover("")
            return
        text = f"x {int(x)}, y {int(y)}"
        hit = self._object_at(event.x, event.y)
        if hit is not None:
            text += f"   #{hit.id} {hit.class_name}, confidence {hit.confidence:.2f}"
        self.on_hover(text)

    def _on_leave(self, event) -> None:
        self._set_placeholder_hover(False)
        self.on_hover("")

    def _set_placeholder_hover(self, hover: bool) -> None:
        if hover != self._hover:
            self._hover = hover
            if self.state.document is None:
                self.refresh()

    def _update_cursor(self) -> None:
        self.canvas.configure(cursor="hand2" if self.state.document is None else "")

    def _object_at(self, sx: float, sy: float) -> DetectedObject | None:
        prediction = self.state.prediction
        if prediction is None:
            return None
        x, y = self.viewport.screen_to_image(sx, sy)
        h, w = prediction.instance_map.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        object_id = int(prediction.instance_map[int(y), int(x)])
        item = prediction.object_by_id(object_id) if object_id > 0 else None
        return item if item is not None and self.state.is_visible(item) else None

    # ---- drawing ---------------------------------------------------------------
    def _base_image(self) -> np.ndarray:
        document, prediction = self.state.document, self.state.prediction
        layer = self.state.display.layer
        artifact = ARTIFACT_OF_LAYER.get(layer)
        if artifact is None or prediction is None:
            return document.display_rgb
        if layer not in self._layer_images:
            try:
                image = to_display_uint8(read_image(prediction.artifacts[artifact]))
                if image.shape[:2] != document.display_rgb.shape[:2]:
                    raise ValueError(f"{artifact} has size {image.shape[:2]}")
                self._layer_images[layer] = image
            except Exception:
                log.exception("could not show layer %s", layer)
                self._layer_images[layer] = document.display_rgb
        return self._layer_images[layer]

    def _class_lookup(self) -> np.ndarray:
        prediction, display = self.state.prediction, self.state.display
        key = (id(prediction), display.hidden_classes, display.min_confidence)
        if key != self._lookup_key:
            size = int(prediction.instance_map.max()) + 1 if prediction.instance_map.size else 1
            self._lookup = class_lookup(prediction.objects, set(display.hidden_classes),
                                        display.min_confidence, size)
            self._lookup_key = key
        return self._lookup

    def _render(self) -> None:
        self._render_job = None
        canvas = self.canvas
        canvas.delete("all")
        self._update_cursor()
        document = self.state.document
        if document is None:
            self._draw_placeholder()
            return
        region = self.viewport.visible_region()
        if region is not None:
            crop = self._base_image()[region.y0:region.y1, region.x0:region.x1]
            zoom = self.viewport.zoom
            interpolation = (cv2.INTER_AREA if zoom < 1 else
                             cv2.INTER_NEAREST if zoom >= 3 else cv2.INTER_LINEAR)
            shown = cv2.resize(crop, (region.screen_w, region.screen_h), interpolation=interpolation)
            prediction, display = self.state.prediction, self.state.display
            if prediction is not None and display.layer == "overlay" and display.show_segments:
                labels = resize_nearest(prediction.instance_map[region.y0:region.y1,
                                                                region.x0:region.x1],
                                        region.screen_w, region.screen_h)
                style = OverlayStyle(display.opacity, display.outlines,
                                     max(1, px(1.5, self.scale)))
                shown = compose_overlay(shown, labels, self._class_lookup(),
                                        colour_table(self.state.classes), style,
                                        highlight_id=self.state.selected_id)
            self._photo = ImageTk.PhotoImage(Image.fromarray(shown))
            canvas.create_image(region.screen_x, region.screen_y, anchor="nw", image=self._photo)
        if self.state.predicting:
            self._draw_badge("Predicting…")

    def _draw_badge(self, text: str) -> None:
        canvas, s = self.canvas, self.scale
        x = canvas.winfo_width() / 2
        label = canvas.create_text(x, px(28, s), text=text, fill="white",
                                   font=(theme.FONT, 12, "bold"))
        x0, y0, x1, y1 = canvas.bbox(label)
        box = canvas.create_rectangle(x0 - px(16, s), y0 - px(8, s), x1 + px(16, s),
                                      y1 + px(8, s), fill=theme.ACCENT, outline="")
        canvas.tag_raise(label, box)

    def _draw_placeholder(self) -> None:
        canvas, u = self.canvas, self.scale
        width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        colour = theme.PLACEHOLDER_HOVER if self._hover else theme.PLACEHOLDER
        cx, cy = width / 2, height / 2 - 50 * u
        x0, y0, x1, y1 = cx - 84 * u, cy - 62 * u, cx + 84 * u, cy + 62 * u
        r, line = 16 * u, max(3, px(4, u))
        frame = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
                 x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        canvas.create_polygon(frame, smooth=True, outline=colour, fill="", width=line)
        canvas.create_oval(x1 - 52 * u, y0 + 18 * u, x1 - 26 * u, y0 + 44 * u,
                           outline=colour, width=line)
        canvas.create_line(x0 + 18 * u, y1 - 18 * u, x0 + 60 * u, y1 - 68 * u, x0 + 92 * u,
                           y1 - 36 * u, x0 + 110 * u, y1 - 56 * u, x1 - 18 * u, y1 - 18 * u,
                           fill=colour, width=line, joinstyle="round", capstyle="round")
        badge = 25 * u
        canvas.create_oval(x1 - badge, y1 - badge, x1 + badge, y1 + badge,
                           fill=theme.ACCENT, outline=theme.CANVAS_BG, width=line + 1)
        arm = 11 * u
        canvas.create_line(x1 - arm, y1, x1 + arm, y1, fill="white", width=line, capstyle="round")
        canvas.create_line(x1, y1 - arm, x1, y1 + arm, fill="white", width=line, capstyle="round")
        canvas.create_text(cx, y1 + 60 * u, text="Insert image", fill=theme.PLACEHOLDER_HOVER,
                           font=(theme.FONT, 16, "bold"))
        canvas.create_text(cx, y1 + 92 * u, fill=theme.PLACEHOLDER, font=(theme.FONT, 10),
                           text="Click to browse  ·  PNG, JPG, BMP, TIFF  ·  Ctrl+O")
