"""The main window's layout: menu bar, toolbar, image view beside the side
panel, the action bar with Clear and Predict, and the status bar.

It only builds widgets and exposes them; every command goes to the controller.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

from v7_segmenter import APP_NAME
from v7_segmenter.services.state import LAYERS, AppState, Topic
from v7_segmenter.settings import Settings
from v7_segmenter.ui.icons import IconSet
from v7_segmenter.ui.image_view import ImageView
from v7_segmenter.ui.side_panel import SidePanel
from v7_segmenter.ui.status_bar import StatusBar
from v7_segmenter.ui.theme import px
from v7_segmenter.ui.tooltip import Tooltip

if TYPE_CHECKING:
    from v7_segmenter.ui.controller import MainController

SIDE_PANEL_WIDTH = 420


class MainWindow:
    def __init__(self, root: tk.Tk, state: AppState, commands: "MainController",
                 icons: IconSet, scale: float, settings: Settings):
        self.root, self.state, self.commands = root, state, commands
        self.icons, self.scale = icons, scale
        self._actions: dict[str, list] = {}
        self.auto_predict = tk.BooleanVar(value=settings.auto_predict)
        self.show_segments = tk.BooleanVar(value=state.display.show_segments)
        self.outlines = tk.BooleanVar(value=state.display.outlines)
        self.layer = tk.StringVar(value=state.display.layer)

        root.title(APP_NAME)
        root.minsize(px(980, scale), px(640, scale))
        self._build_menu()
        self._build_toolbar()
        ttk.Separator(root, orient="horizontal").pack(side="top", fill="x")
        self.status_bar = StatusBar(root, state, scale)
        self.status_bar.pack(side="bottom", fill="x")
        ttk.Separator(root, orient="horizontal").pack(side="bottom", fill="x")
        self._build_action_bar()
        ttk.Separator(root, orient="horizontal").pack(side="bottom", fill="x")
        self._build_body(settings)

        bus = state.bus
        bus.subscribe(Topic.DOCUMENT, self._refresh_info)
        bus.subscribe(Topic.DISPLAY, self._refresh_display_vars)
        self.refresh_recent(settings.recent_files)
        self._refresh_info()

    # ---- construction ----------------------------------------------------------
    def _register(self, action: str, item) -> None:
        self._actions.setdefault(action, []).append(item)

    def _menu_command(self, menu: tk.Menu, action: str | None, label: str, command,
                      accelerator: str = "") -> None:
        menu.add_command(label=label, command=command, accelerator=accelerator)
        if action:
            self._register(action, (menu, menu.index("end")))

    def _build_menu(self) -> None:
        c = self.commands
        bar = tk.Menu(self.root)

        file_menu = tk.Menu(bar, tearoff=False)
        self._menu_command(file_menu, "open", "Insert image…", c.open_image_dialog, "Ctrl+O")
        self.recent_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Recent images", menu=self.recent_menu)
        self._register("open", (file_menu, file_menu.index("end")))
        file_menu.add_separator()
        self._menu_command(file_menu, "previous", "Previous image in folder", c.previous_image, "PgUp")
        self._menu_command(file_menu, "next", "Next image in folder", c.next_image, "PgDn")
        file_menu.add_separator()
        self._menu_command(file_menu, "save", "Save overlay image…", c.save_overlay_dialog, "Ctrl+S")
        self._menu_command(file_menu, "export", "Export all results…", c.export_results_dialog, "Ctrl+E")
        file_menu.add_separator()
        self._menu_command(file_menu, None, "Exit", c.quit, "Alt+F4")
        bar.add_cascade(label="File", menu=file_menu)

        model_menu = tk.Menu(bar, tearoff=False)
        self._menu_command(model_menu, "predict", "Predict", c.predict, "F5")
        self._menu_command(model_menu, "clear", "Clear", c.clear, "Delete")
        model_menu.add_separator()
        model_menu.add_checkbutton(label="Predict automatically when an image is opened",
                                   variable=self.auto_predict, command=c.auto_predict_changed)
        bar.add_cascade(label="Model", menu=model_menu)

        view_menu = tk.Menu(bar, tearoff=False)
        self._menu_command(view_menu, "zoom", "Zoom in", c.zoom_in, "Ctrl++")
        self._menu_command(view_menu, "zoom", "Zoom out", c.zoom_out, "Ctrl+-")
        self._menu_command(view_menu, "zoom", "Fit to window", c.fit, "Ctrl+0")
        self._menu_command(view_menu, "zoom", "Actual size", c.actual_size, "Ctrl+1")
        view_menu.add_separator()
        view_menu.add_checkbutton(label="Show segments", variable=self.show_segments,
                                  command=c.toggle_segments, accelerator="Ctrl+H")
        view_menu.add_checkbutton(label="Outlines", variable=self.outlines,
                                  command=c.toggle_outlines, accelerator="Ctrl+T")
        view_menu.add_separator()
        for key, text, artifact in LAYERS:
            view_menu.add_radiobutton(label=text, value=key, variable=self.layer,
                                      command=lambda k=key: c.set_layer(k))
            if artifact is not None:
                self._register("layers", (view_menu, view_menu.index("end")))
        bar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(bar, tearoff=False)
        self._menu_command(help_menu, None, "Keyboard shortcuts", c.show_shortcuts, "F1")
        self._menu_command(help_menu, None, "Open the log folder", c.open_log_folder)
        help_menu.add_separator()
        self._menu_command(help_menu, None, f"About {APP_NAME}", c.show_about)
        bar.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=bar)

    def _tool(self, parent, action: str, icon: str, text: str, command, tip: str) -> ttk.Button:
        button = ttk.Button(parent, text=text, image=self.icons.button_image(icon),
                            compound="left" if text else "image", style="Toolbar.TButton",
                            command=command)
        if not self.icons.button_image(icon):
            button.configure(text=text or tip.split(" (")[0], compound="text")
        button.pack(side="left", padx=(0, px(2, self.scale)))
        Tooltip(button, tip)
        self._register(action, button)
        return button

    def _build_toolbar(self) -> None:
        c, s = self.commands, self.scale
        bar = ttk.Frame(self.root, padding=(px(8, s), px(5, s)))
        bar.pack(side="top", fill="x")
        self._tool(bar, "open", "open", "Insert image", c.open_image_dialog, "Insert an image (Ctrl+O)")
        self._tool(bar, "previous", "previous", "", c.previous_image, "Previous image in the folder (PgUp)")
        self._tool(bar, "next", "next", "", c.next_image, "Next image in the folder (PgDn)")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=px(8, s))
        self._tool(bar, "zoom", "zoom_out", "", c.zoom_out, "Zoom out (Ctrl+minus)")
        self._tool(bar, "zoom", "zoom_in", "", c.zoom_in, "Zoom in (Ctrl+plus)")
        self._tool(bar, "zoom", "fit", "Fit", c.fit, "Fit the image to the window (Ctrl+0)")
        self._tool(bar, "zoom", "", "100%", c.actual_size, "Actual size (Ctrl+1)")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=px(8, s))
        self._tool(bar, "save", "save", "Save overlay", c.save_overlay_dialog,
                   "Save the photo with its segments as an image (Ctrl+S)")
        self._tool(bar, "export", "export", "Export", c.export_results_dialog,
                   "Export masks, heatmaps, object list and overlay to a folder (Ctrl+E)")
        auto = ttk.Checkbutton(bar, text="Predict automatically", variable=self.auto_predict,
                               command=c.auto_predict_changed)
        auto.pack(side="right")
        Tooltip(auto, "Run the model as soon as an image is opened")

    def _build_body(self, settings: Settings) -> None:
        self.paned = ttk.Panedwindow(self.root, orient="horizontal")
        self.paned.pack(side="top", fill="both", expand=True)
        self.image_view = ImageView(self.paned, self.state, self.scale)
        self.side_panel = SidePanel(self.paned, self.state, self.scale)
        self.paned.add(self.image_view, weight=1)
        self.paned.add(self.side_panel, weight=0)
        width = settings.side_panel_width or px(SIDE_PANEL_WIDTH, self.scale)
        self.root.after(150, lambda: self._place_sash(width))

    def _place_sash(self, side_width: int) -> None:
        total = self.paned.winfo_width()
        if total > side_width + 200:
            self.paned.sashpos(0, total - side_width)

    def side_panel_width(self) -> int:
        return max(0, self.paned.winfo_width() - self.paned.sashpos(0))

    def _build_action_bar(self) -> None:
        c, s = self.commands, self.scale
        bar = ttk.Frame(self.root, padding=(px(12, s), px(8, s)))
        bar.pack(side="bottom", fill="x")
        self.info = ttk.Label(bar, text="", style="Info.TLabel")
        self.info.pack(side="left")
        predict = ttk.Button(bar, text="Predict", style="Accent.TButton",
                             image=self.icons.button_image("predict"), compound="left",
                             command=c.predict)
        predict.pack(side="right")
        clear = ttk.Button(bar, text="Clear", style="Action.TButton",
                           image=self.icons.button_image("clear"), compound="left",
                           command=c.clear)
        clear.pack(side="right", padx=(0, px(10, s)))
        Tooltip(predict, "Run the V7 model on this image (F5)")
        Tooltip(clear, "Remove the image and its prediction (Delete)")
        self._register("predict", predict)
        self._register("clear", clear)

    # ---- updates -------------------------------------------------------------------
    def set_enabled(self, action: str, enabled: bool) -> None:
        for item in self._actions.get(action, []):
            if isinstance(item, tuple):
                menu, index = item
                menu.entryconfigure(index, state="normal" if enabled else "disabled")
            else:
                item.state(["!disabled"] if enabled else ["disabled"])

    def refresh_recent(self, recent: list[str]) -> None:
        menu = self.recent_menu
        menu.delete(0, "end")
        if not recent:
            menu.add_command(label="(none yet)", state="disabled")
            return
        for number, text in enumerate(recent, start=1):
            path = Path(text)
            label = f"{number}  {path.name}    —    {path.parent}"
            menu.add_command(label=label, underline=0,
                             command=lambda p=path: self.commands.open_image(p))
        menu.add_separator()
        menu.add_command(label="Clear the list", command=self.commands.clear_recent)

    def _refresh_info(self) -> None:
        document = self.state.document
        if document is None:
            self.info.configure(text="No image inserted")
            self.root.title(APP_NAME)
            return
        text = f"{document.name}   ·   {document.width} × {document.height}"
        if self.state.folder_position:
            index, total = self.state.folder_position
            text += f"   ·   image {index} of {total} in the folder"
        self.info.configure(text=text)
        self.root.title(f"{document.name} — {APP_NAME}")

    def _refresh_display_vars(self) -> None:
        display = self.state.display
        self.show_segments.set(display.show_segments)
        self.outlines.set(display.outlines)
        self.layer.set(display.layer)
