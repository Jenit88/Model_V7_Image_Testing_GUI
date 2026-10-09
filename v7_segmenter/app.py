"""Composition root: builds the services and the window, wires them, runs Tk.

    python run_gui.py [IMAGE]                       start, optionally opening IMAGE
    python run_gui.py --demo IMAGE --screenshots P  scripted check (see demo.py)
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import logging
import re
import sys
import tkinter as tk
from pathlib import Path

from PIL import ImageTk

from v7_segmenter import APP_NAME, __version__
from v7_segmenter.events import EventBus
from v7_segmenter.inference.engine import V7Engine
from v7_segmenter.inference.worker import BackgroundWorker
from v7_segmenter.logging_setup import configure_logging
from v7_segmenter.paths import default_paths
from v7_segmenter.services.navigation import FolderNavigator
from v7_segmenter.services.state import AppState, DisplayOptions
from v7_segmenter.services.workspace import Workspace
from v7_segmenter.settings import Settings
from v7_segmenter.ui.controller import MainController
from v7_segmenter.ui.icons import IconSet, app_icon
from v7_segmenter.ui.main_window import MainWindow
from v7_segmenter.ui.theme import apply_theme, px, ui_scale

log = logging.getLogger(__name__)


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run_gui.py", description=APP_NAME)
    parser.add_argument("image", nargs="?", help="image to open at start")
    parser.add_argument("--demo", metavar="IMAGE", help="run the scripted check on IMAGE")
    parser.add_argument("--screenshots", metavar="PATH",
                        help="with --demo: save screenshots as PATH_<step>.png")
    return parser.parse_args(argv)


def _enable_dpi_awareness() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def _work_area(root: tk.Tk) -> tuple[int, int, int, int]:
    """The desktop minus the taskbar, as (left, top, width, height)."""
    if sys.platform == "win32":
        rect = ctypes.wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def _initial_geometry(root: tk.Tk, settings: Settings, scale: float) -> str:
    left, top, screen_w, screen_h = _work_area(root)
    match = re.fullmatch(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", settings.window_geometry or "")
    if match:
        w, h, x, y = (int(v) for v in match.groups())
        if 400 < w <= screen_w and 300 < h <= screen_h and -50 < x < screen_w - 100 \
                and 0 <= y < screen_h - 100:
            return settings.window_geometry
    # Leave room for the title bar, menu bar and borders, which Tk does not count.
    w = min(px(1440, scale), int(screen_w * 0.94))
    h = min(px(900, scale), screen_h - px(90, scale))
    return f"{w}x{h}+{left + (screen_w - w) // 2}+{top + max(0, (screen_h - h - px(70, scale)) // 2)}"


def run(argv: list[str] | None = None) -> None:
    args = _arguments(sys.argv[1:] if argv is None else argv)
    paths = default_paths()
    configure_logging(paths.log_dir)
    log.info("starting %s %s from %s", APP_NAME, __version__, paths.root)
    _enable_dpi_awareness()

    root = tk.Tk()
    root.withdraw()
    scale = ui_scale(root)
    apply_theme(root, scale)
    icon = ImageTk.PhotoImage(app_icon(64))
    root.iconphoto(True, icon)

    settings = Settings.load(paths.settings_file)
    state = AppState(EventBus())
    state.display = DisplayOptions(show_segments=settings.show_segments,
                                   opacity=settings.opacity, outlines=settings.show_outlines)
    controller = MainController(
        root, state, V7Engine(paths), BackgroundWorker(root), Workspace(paths.workspace_root),
        FolderNavigator(), settings, paths, scale)
    window = MainWindow(root, state, controller, IconSet(scale), scale, settings)
    controller.attach(window)

    root.geometry(_initial_geometry(root, settings, scale))
    root.deiconify()
    if settings.window_maximized:
        root.state("zoomed")
    controller.start()
    if args.image:
        root.after(300, lambda: controller.open_image(Path(args.image)))
    if args.demo:
        from v7_segmenter.demo import run_demo
        run_demo(controller, window, Path(args.demo),
                 Path(args.screenshots) if args.screenshots else None)
    root.mainloop()
