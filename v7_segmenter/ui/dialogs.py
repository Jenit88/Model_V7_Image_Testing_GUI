"""About and keyboard-shortcut windows, and error reporting."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from v7_segmenter import APP_NAME, __version__
from v7_segmenter.domain import ModelInfo
from v7_segmenter.ui import theme
from v7_segmenter.ui.theme import px

SHORTCUTS = (
    ("Ctrl+O", "Insert (open) an image"),
    ("F5  or  Ctrl+Enter", "Predict"),
    ("Delete", "Clear the image and its prediction"),
    ("Page Up / Alt+Left", "Previous image in the folder"),
    ("Page Down / Alt+Right", "Next image in the folder"),
    ("Ctrl+S", "Save the overlay image"),
    ("Ctrl+E", "Export all results to a folder"),
    ("Ctrl+H", "Show / hide segments"),
    ("Ctrl+T", "Outlines on / off"),
    ("Ctrl+plus / Ctrl+minus", "Zoom in / out (or the mouse wheel)"),
    ("Ctrl+0", "Fit the image to the window"),
    ("Ctrl+1", "Actual size (100%)"),
    ("Drag", "Pan a zoomed image"),
    ("Click an object", "Highlight it and select it in the table"),
    ("Esc", "Deselect"),
    ("F1", "This list"),
)


def show_error(parent: tk.Misc, title: str, message: str, detail: str = "") -> None:
    messagebox.showerror(title, message, detail=detail, parent=parent)


def _dialog(parent: tk.Misc, title: str, scale: float) -> tuple[tk.Toplevel, ttk.Frame]:
    window = tk.Toplevel(parent)
    window.title(title)
    window.transient(parent)
    window.resizable(False, False)
    body = ttk.Frame(window, padding=px(18, scale))
    body.pack(fill="both", expand=True)
    window.bind("<Escape>", lambda e: window.destroy())
    return window, body


def _finish(window: tk.Toplevel, body: ttk.Frame, scale: float) -> None:
    ttk.Button(body, text="Close", command=window.destroy).grid(
        row=99, column=0, columnspan=2, sticky="e", pady=(px(14, scale), 0))
    window.update_idletasks()
    parent = window.master
    x = parent.winfo_rootx() + (parent.winfo_width() - window.winfo_width()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - window.winfo_height()) // 3
    window.geometry(f"+{max(0, x)}+{max(0, y)}")
    window.grab_set()
    window.focus_set()


def show_shortcuts(parent: tk.Misc, scale: float) -> None:
    window, body = _dialog(parent, "Keyboard shortcuts", scale)
    for row, (keys, action) in enumerate(SHORTCUTS):
        ttk.Label(body, text=keys, font=(theme.FONT, 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=(0, px(18, scale)), pady=1)
        ttk.Label(body, text=action).grid(row=row, column=1, sticky="w", pady=1)
    _finish(window, body, scale)


def show_about(parent: tk.Misc, scale: float, model: ModelInfo | None, model_error: str | None) -> None:
    window, body = _dialog(parent, f"About {APP_NAME}", scale)
    ttk.Label(body, text=APP_NAME, font=(theme.FONT, 16, "bold")).grid(row=0, column=0,
                                                                     columnspan=2, sticky="w")
    ttk.Label(body, text=f"Version {__version__}  ·  PCB instance segmentation with the V7 model",
              style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w",
                                         pady=(0, px(12, scale)))
    rows: list[tuple[str, str]] = []
    if model is None:
        rows.append(("Model", model_error or "still loading"))
    else:
        card = model.card
        rows += [
            ("Model", model.name),
            ("File", model.file.name),
            ("SHA-256", model.sha256_prefix + "…"),
            ("Runs on", f"{model.device} (loaded in {model.load_seconds:.0f} s)"),
        ]
        if card.get("epoch") is not None:
            rows.append(("Training epoch", str(card["epoch"])))
        for key, label in (("validation", "Validation"), ("test", "Test")):
            scores = card.get(key)
            if isinstance(scores, dict) and "mask_map50_95" in scores:
                rows.append((f"{label} mAP50-95",
                             f"{scores['mask_map50_95'] * 100:.1f}%  on {scores.get('images', '?')} images"))
        if card.get("source"):
            rows.append(("Source", str(card["source"])))
    rows.append(("Inference code", "model_v7.py, used read-only (predict_one_image)"))
    for row, (label, value) in enumerate(rows, start=2):
        ttk.Label(body, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="nw",
                                                              padx=(0, px(18, scale)), pady=1)
        ttk.Label(body, text=value, wraplength=px(420, scale), justify="left").grid(
            row=row, column=1, sticky="w", pady=1)
    _finish(window, body, scale)
