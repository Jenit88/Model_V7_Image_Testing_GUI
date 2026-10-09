"""Hover tooltips for toolbar buttons."""
from __future__ import annotations

import tkinter as tk

from v7_segmenter.ui import theme


class Tooltip:
    DELAY_MS = 450

    def __init__(self, widget: tk.Misc, text: str):
        self.widget, self.text = widget, text
        self._job = None
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, event=None) -> None:
        self._cancel()
        self._job = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self) -> None:
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None

    def _show(self) -> None:
        self._job = None
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        tk.Label(window, text=self.text, background="#202124", foreground="white",
                 font=(theme.FONT, 9), padx=8, pady=4).pack()
        self._window = window

    def _hide(self, event=None) -> None:
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None
