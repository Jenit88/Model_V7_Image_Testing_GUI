"""File logging, including for pythonw where there is no console at all."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_NAME = "v7_segmenter.log"


class _LogStream:
    """Stands in for sys.stdout/sys.stderr under pythonw, where both are None
    and a library writing to them directly would raise."""

    def __init__(self, level: int):
        self.level = level
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                logging.getLogger("console").log(self.level, line)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def configure_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_NAME
    handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    if sys.stdout is None:
        sys.stdout = _LogStream(logging.INFO)
    if sys.stderr is None:
        sys.stderr = _LogStream(logging.WARNING)

    def log_uncaught(kind, value, traceback):
        logging.getLogger("uncaught").critical("unhandled exception", exc_info=(kind, value, traceback))

    sys.excepthook = log_uncaught
    return log_file
