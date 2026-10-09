"""Bottom status bar: message, progress, cursor position, zoom and model state."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from v7_segmenter.services.state import AppState, ModelStatus, Topic
from v7_segmenter.ui import theme
from v7_segmenter.ui.theme import px


class StatusBar(ttk.Frame):
    def __init__(self, master: tk.Misc, state: AppState, scale: float):
        super().__init__(master, padding=(px(10, scale), px(3, scale)))
        self.state = state
        self.scale = scale
        self.message = ttk.Label(self, text="", style="Muted.TLabel")
        self.message.pack(side="left")
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=px(130, scale))
        self.model = tk.Label(self, text="", font=(theme.FONT, 9), anchor="e")
        self.model.pack(side="right")
        ttk.Separator(self, orient="vertical").pack(side="right", fill="y", padx=px(8, scale))
        self.zoom = ttk.Label(self, text="", width=7, anchor="e", style="Muted.TLabel")
        self.zoom.pack(side="right")
        ttk.Separator(self, orient="vertical").pack(side="right", fill="y", padx=px(8, scale))
        self.cursor = ttk.Label(self, text="", style="Muted.TLabel")
        self.cursor.pack(side="right")
        self._progress_shown = False
        state.bus.subscribe(Topic.ACTIVITY, self._refresh)
        state.bus.subscribe(Topic.MODEL, self._refresh)
        self._refresh()

    def set_cursor(self, text: str) -> None:
        self.cursor.configure(text=text)

    def set_zoom(self, zoom: float | None) -> None:
        self.zoom.configure(text="" if zoom is None else f"{zoom * 100:.0f}%")

    def _refresh(self) -> None:
        state = self.state
        self.message.configure(text=state.message)
        busy = state.predicting or state.model_status is ModelStatus.LOADING
        if busy and not self._progress_shown:
            self.progress.pack(side="left", padx=(px(10, self.scale), 0))
            self.progress.start(12)
        elif not busy and self._progress_shown:
            self.progress.stop()
            self.progress.pack_forget()
        self._progress_shown = busy
        if state.model_status is ModelStatus.LOADING:
            self.model.configure(text="●  Loading model…", fg=theme.STATUS_BUSY)
        elif state.model_status is ModelStatus.FAILED:
            self.model.configure(text="●  Model not available", fg=theme.STATUS_ERROR)
        else:
            info = state.model_info
            self.model.configure(text=f"●  {info.name} · {info.device}", fg=theme.STATUS_OK)
