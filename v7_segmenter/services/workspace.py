"""Scratch folders for model_v7's per-prediction output files.

predict_one_image saves its outputs (masks, heatmaps, JSON) to a folder; the
application keeps the most recent ones in the system temp folder for display
and export, and deletes them on exit. Sessions left behind by a crash are
removed the next day.
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)


class Workspace:
    KEEP_PREDICTIONS = 12
    STALE_AFTER_S = 24 * 3600

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self._remove_stale_sessions(root)
        self.session = Path(tempfile.mkdtemp(prefix="session-", dir=root))
        self._folders: list[Path] = []
        self._counter = 0

    def new_prediction_dir(self, stem: str) -> Path:
        self._counter += 1
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:60] or "image"
        folder = self.session / f"{self._counter:04d}_{safe}"
        folder.mkdir(parents=True)
        self._folders.append(folder)
        while len(self._folders) > self.KEEP_PREDICTIONS:
            shutil.rmtree(self._folders.pop(0), ignore_errors=True)
        return folder

    def close(self) -> None:
        shutil.rmtree(self.session, ignore_errors=True)

    def _remove_stale_sessions(self, root: Path) -> None:
        cutoff = time.time() - self.STALE_AFTER_S
        for folder in root.glob("session-*"):
            try:
                if folder.stat().st_mtime < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
            except OSError:
                log.warning("could not inspect old workspace %s", folder)
