"""User preferences, kept between sessions in config/settings.json."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import ClassVar

log = logging.getLogger(__name__)


@dataclass
class Settings:
    window_geometry: str = ""          # the normal (not maximised) size and position
    window_maximized: bool = True
    side_panel_width: int = 0
    last_folder: str = ""
    recent_files: list[str] = field(default_factory=list)
    opacity: float = 0.45
    show_segments: bool = True
    show_outlines: bool = True
    auto_predict: bool = False

    MAX_RECENT: ClassVar[int] = 10

    def add_recent(self, path: Path | str) -> None:
        text = str(path)
        self.recent_files = [text] + [p for p in self.recent_files
                                      if os.path.normcase(p) != os.path.normcase(text)]
        del self.recent_files[self.MAX_RECENT:]

    def remove_recent(self, path: Path | str) -> None:
        text = os.path.normcase(str(path))
        self.recent_files = [p for p in self.recent_files if os.path.normcase(p) != text]

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Read settings; a missing, unreadable or partly invalid file never
        stops the application -- unknown keys are ignored, bad values reset."""
        settings = cls()
        if not path.is_file():
            return settings
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            log.warning("ignoring unreadable settings file %s: %s", path, error)
            return settings
        if not isinstance(data, dict):
            return settings
        for item in fields(cls):
            if item.name not in data:
                continue
            value, default = data[item.name], getattr(settings, item.name)
            if isinstance(default, bool):
                ok = isinstance(value, bool)
            elif isinstance(default, (int, float)):
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif isinstance(default, list):
                ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
            else:
                ok = isinstance(value, type(default))
            if ok:
                setattr(settings, item.name, value)
        settings.opacity = min(1.0, max(0.0, float(settings.opacity)))
        del settings.recent_files[cls.MAX_RECENT:]
        return settings

    def save(self, path: Path) -> None:
        """Write atomically, so a crash mid-write cannot corrupt the file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(temporary, path)
