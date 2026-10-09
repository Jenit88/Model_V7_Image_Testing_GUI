"""Toolbar icons drawn from Windows' own icon font, with a text-only fallback."""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageTk

log = logging.getLogger(__name__)

FONT_FILES = (
    Path(r"C:\Windows\Fonts\SegoeIcons.ttf"),    # Segoe Fluent Icons (Windows 11)
    Path(r"C:\Windows\Fonts\segmdl2.ttf"),       # Segoe MDL2 Assets (Windows 10)
)
GLYPHS = {
    "open": "\uE8E5",
    "previous": "\uE892",
    "next": "\uE893",
    "zoom_in": "\uE8A3",
    "zoom_out": "\uE71F",
    "fit": "\uE9A6",
    "save": "\uE74E",
    "export": "\uEDE1",
    "predict": "\uE768",
    "clear": "\uE894",
    "info": "\uE946",
}
NORMAL = (32, 33, 36, 255)
DISABLED = (160, 160, 160, 255)


def app_icon(size: int = 64) -> Image.Image:
    """Window icon: a dark board with one rectangle and one round pad."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    s = size / 64
    draw.rounded_rectangle((2 * s, 2 * s, 62 * s, 62 * s), radius=12 * s, fill=(30, 31, 34, 255))
    draw.rounded_rectangle((11 * s, 13 * s, 35 * s, 33 * s), radius=3 * s,
                           fill=(230, 65, 65, 255))
    draw.ellipse((33 * s, 31 * s, 54 * s, 52 * s), fill=(65, 135, 230, 255))
    draw.ellipse((13 * s, 40 * s, 25 * s, 52 * s), fill=(60, 180, 90, 255))
    return image


class IconSet:
    def __init__(self, scale: float):
        self.size = max(14, int(round(16 * scale)))
        self._font = None
        for path in FONT_FILES:
            if path.is_file():
                try:
                    self._font = ImageFont.truetype(str(path), self.size)
                    break
                except OSError:
                    log.warning("could not load icon font %s", path)
        self._cache: dict[tuple[str, tuple], ImageTk.PhotoImage] = {}

    def _photo(self, name: str, colour: tuple) -> ImageTk.PhotoImage | None:
        if self._font is None or name not in GLYPHS:
            return None
        key = (name, colour)
        if key not in self._cache:
            box = self.size + 4
            image = Image.new("RGBA", (box, box), (0, 0, 0, 0))
            ImageDraw.Draw(image).text((box / 2, box / 2), GLYPHS[name], font=self._font,
                                       fill=colour, anchor="mm")
            self._cache[key] = ImageTk.PhotoImage(image)
        return self._cache[key]

    def button_image(self, name: str):
        """Value for a ttk widget's `image` option (greyed when disabled), or ''."""
        normal, disabled = self._photo(name, NORMAL), self._photo(name, DISABLED)
        if normal is None:
            return ""
        return (normal, "disabled", disabled)
