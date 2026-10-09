"""Zoom and pan: which part of the image is visible, and where on screen.

Pure arithmetic, so it is unit-tested without a window. Screen coordinates are
canvas pixels; image coordinates are pixels of the photo at full resolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VisibleRegion:
    x0: int     # image pixels [x0, x1) x [y0, y1) are visible ...
    y0: int
    x1: int
    y1: int
    screen_x: int   # ... drawn with their top-left corner here,
    screen_y: int
    screen_w: int   # at this size on screen
    screen_h: int


class Viewport:
    MIN_ZOOM = 0.02
    MAX_ZOOM = 32.0

    def __init__(self) -> None:
        self.image_w = self.image_h = 0
        self.view_w = self.view_h = 1
        self.zoom = 1.0
        self.center_x = self.center_y = 0.0
        self.fitted = True

    # ---- configuration ---------------------------------------------------
    def set_image(self, width: int, height: int) -> None:
        self.image_w, self.image_h = int(width), int(height)
        self.fit()

    def set_view(self, width: int, height: int) -> None:
        self.view_w, self.view_h = max(1, int(width)), max(1, int(height))
        if self.fitted:
            self.fit()
        else:
            self._clamp()

    @property
    def has_image(self) -> bool:
        return self.image_w > 0 and self.image_h > 0

    # ---- zoom and pan ------------------------------------------------------
    def fit_zoom(self, margin: int = 12) -> float:
        if not self.has_image:
            return 1.0
        usable_w = max(1, self.view_w - 2 * margin)
        usable_h = max(1, self.view_h - 2 * margin)
        return min(usable_w / self.image_w, usable_h / self.image_h, self.MAX_ZOOM)

    def fit(self) -> None:
        self.zoom = max(self.MIN_ZOOM, self.fit_zoom())
        self.center_x, self.center_y = self.image_w / 2, self.image_h / 2
        self.fitted = True

    def set_zoom(self, zoom: float, anchor: tuple[float, float] | None = None) -> None:
        """Zoom so the image point under `anchor` (screen) stays where it is."""
        if not self.has_image:
            return
        if anchor is None:
            anchor = (self.view_w / 2, self.view_h / 2)
        before = self.screen_to_image(*anchor)
        self.zoom = min(self.MAX_ZOOM, max(self.MIN_ZOOM, zoom))
        after = self.screen_to_image(*anchor)
        self.center_x += before[0] - after[0]
        self.center_y += before[1] - after[1]
        self.fitted = False
        self._clamp()

    def zoom_by(self, factor: float, anchor: tuple[float, float] | None = None) -> None:
        self.set_zoom(self.zoom * factor, anchor)

    def pan(self, dx: float, dy: float) -> None:
        """Move the image by (dx, dy) screen pixels."""
        self.center_x -= dx / self.zoom
        self.center_y -= dy / self.zoom
        self.fitted = False
        self._clamp()

    def center_on(self, x: float, y: float) -> None:
        self.center_x, self.center_y = x, y
        self.fitted = False
        self._clamp()

    def _clamp(self) -> None:
        self.center_x = min(max(self.center_x, 0.0), float(self.image_w))
        self.center_y = min(max(self.center_y, 0.0), float(self.image_h))

    # ---- coordinate conversion -------------------------------------------
    def _origin(self) -> tuple[float, float]:
        """Image coordinates of the canvas's top-left corner."""
        return (self.center_x - self.view_w / (2 * self.zoom),
                self.center_y - self.view_h / (2 * self.zoom))

    def screen_to_image(self, sx: float, sy: float) -> tuple[float, float]:
        left, top = self._origin()
        return left + sx / self.zoom, top + sy / self.zoom

    def image_to_screen(self, x: float, y: float) -> tuple[float, float]:
        left, top = self._origin()
        return (x - left) * self.zoom, (y - top) * self.zoom

    def contains_box(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        sx0, sy0 = self.image_to_screen(x0, y0)
        sx1, sy1 = self.image_to_screen(x1, y1)
        return sx0 >= 0 and sy0 >= 0 and sx1 <= self.view_w and sy1 <= self.view_h

    def visible_region(self) -> VisibleRegion | None:
        if not self.has_image:
            return None
        left, top = self._origin()
        x0 = max(0, math.floor(left))
        y0 = max(0, math.floor(top))
        x1 = min(self.image_w, math.ceil(left + self.view_w / self.zoom))
        y1 = min(self.image_h, math.ceil(top + self.view_h / self.zoom))
        if x1 <= x0 or y1 <= y0:
            return None
        sx, sy = self.image_to_screen(x0, y0)
        return VisibleRegion(
            x0, y0, x1, y1,
            screen_x=int(round(sx)), screen_y=int(round(sy)),
            screen_w=max(1, int(round((x1 - x0) * self.zoom))),
            screen_h=max(1, int(round((y1 - y0) * self.zoom))),
        )
