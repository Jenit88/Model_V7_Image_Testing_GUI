"""Locations of everything the application reads and writes.

The application only ever reads model_v7_resource/. It writes to config/
(settings), logs/ and a scratch workspace in the system temp folder.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @property
    def resource_dir(self) -> Path:
        return self.root / "model_v7_resource"

    @property
    def model_file(self) -> Path:
        return self.resource_dir / "best_fine_tuned_model_v7_instance.keras"

    @property
    def model_code(self) -> Path:
        return self.resource_dir / "model_v7.py"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def model_card(self) -> Path:
        return self.config_dir / "model_card.json"

    @property
    def settings_file(self) -> Path:
        return self.config_dir / "settings.json"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    @property
    def workspace_root(self) -> Path:
        return Path(tempfile.gettempdir()) / "v7_segmenter"


def default_paths() -> AppPaths:
    """Paths relative to this installation (the folder holding v7_segmenter/)."""
    return AppPaths(root=Path(__file__).resolve().parent.parent)
