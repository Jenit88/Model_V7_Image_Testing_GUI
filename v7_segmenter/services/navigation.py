"""Previous / next photo in the folder of the current one."""
from __future__ import annotations

import os
import re
from pathlib import Path

from v7_segmenter.imaging.io import is_image_file


def natural_key(path: Path) -> list:
    """Sort 'img2' before 'img10'."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.name.lower())]


class FolderNavigator:
    def __init__(self) -> None:
        self._files: list[Path] = []
        self._index = -1

    def set_current(self, path: Path) -> None:
        try:
            files = [p for p in path.parent.iterdir() if p.is_file() and is_image_file(p)]
        except OSError:
            files = []
        files.sort(key=natural_key)
        wanted = os.path.normcase(str(path))
        index = next((i for i, p in enumerate(files) if os.path.normcase(str(p)) == wanted), -1)
        if index < 0:               # e.g. an unusual extension: still navigable from here
            files.append(path)
            files.sort(key=natural_key)
            index = files.index(path)
        self._files, self._index = files, index

    def clear(self) -> None:
        self._files, self._index = [], -1

    @property
    def position(self) -> tuple[int, int] | None:
        return (self._index + 1, len(self._files)) if self._index >= 0 else None

    @property
    def has_previous(self) -> bool:
        return self._index > 0

    @property
    def has_next(self) -> bool:
        return 0 <= self._index < len(self._files) - 1

    def peek(self, step: int) -> Path | None:
        index = self._index + step
        return self._files[index] if 0 <= index < len(self._files) and self._index >= 0 else None
