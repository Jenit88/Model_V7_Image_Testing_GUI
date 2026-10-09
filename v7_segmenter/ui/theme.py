"""Colours, fonts and ttk styles."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

CANVAS_BG = "#1e1f22"
PLACEHOLDER = "#9aa0a6"
PLACEHOLDER_HOVER = "#e8eaed"
ACCENT = "#2563eb"
MUTED = "#5f6368"
STATUS_OK = "#16a34a"
STATUS_BUSY = "#d97706"
STATUS_ERROR = "#dc2626"
FONT = "Segoe UI"


def ui_scale(root: tk.Misc) -> float:
    """Screen scaling relative to 96 dpi; pixel sizes are multiplied by it."""
    return max(1.0, float(root.winfo_fpixels("1i")) / 96.0)


def px(value: float, scale: float) -> int:
    return int(round(value * scale))


def apply_theme(root: tk.Misc, scale: float) -> ttk.Style:
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    style.configure("Accent.TButton", font=(FONT, 10, "bold"), padding=(px(18, scale), px(5, scale)))
    style.map("Accent.TButton", foreground=[("disabled", "#a0a0a0")])
    style.configure("Action.TButton", font=(FONT, 10), padding=(px(18, scale), px(5, scale)))
    style.configure("Toolbar.TButton", padding=(px(6, scale), px(3, scale)))
    style.configure("Heading.TLabel", font=(FONT, 10, "bold"))
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Info.TLabel", font=(FONT, 10))
    style.configure("Treeview", rowheight=px(22, scale))
    style.configure("Section.TLabelframe.Label", font=(FONT, 10, "bold"))
    return style
