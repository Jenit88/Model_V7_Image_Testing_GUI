"""The controller: turns user actions into service calls and state changes.

Views never change the state themselves; they call methods here. Slow work
(loading the model, predicting) goes to the background worker, and its result
is applied back on the Tk thread.
"""
from __future__ import annotations

import logging
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

from v7_segmenter.domain import ImageDocument, Prediction
from v7_segmenter.imaging.io import FILE_DIALOG_TYPES, read_image, to_display_uint8
from v7_segmenter.imaging.render import resize_nearest
from v7_segmenter.inference.engine import V7Engine
from v7_segmenter.inference.worker import BackgroundWorker
from v7_segmenter.paths import AppPaths
from v7_segmenter.services import export
from v7_segmenter.services.navigation import FolderNavigator
from v7_segmenter.services.state import LAYERS, AppState, Topic
from v7_segmenter.services.workspace import Workspace
from v7_segmenter.settings import Settings
from v7_segmenter.ui import dialogs

if TYPE_CHECKING:
    from v7_segmenter.ui.main_window import MainWindow

log = logging.getLogger(__name__)
ARTIFACT_LAYERS = {key for key, _, artifact in LAYERS if artifact is not None}


class MainController:
    def __init__(self, root: tk.Tk, state: AppState, engine: V7Engine, worker: BackgroundWorker,
                 workspace: Workspace, navigator: FolderNavigator, settings: Settings,
                 paths: AppPaths, scale: float):
        self.root, self.state, self.engine, self.worker = root, state, engine, worker
        self.workspace, self.navigator, self.settings = workspace, navigator, settings
        self.paths, self.scale = paths, scale
        self.window: MainWindow | None = None
        self._generation = 0          # bumps whenever the photo changes

    # ---- wiring ----------------------------------------------------------------
    def attach(self, window: "MainWindow") -> None:
        self.window = window
        view, panel = window.image_view, window.side_panel
        view.on_insert = self.open_image_dialog
        view.on_select = self.select_object
        view.on_hover = window.status_bar.set_cursor
        view.on_zoom = window.status_bar.set_zoom
        panel.on_class_toggle = self.set_class_visible
        panel.on_display = self.update_display
        panel.on_select = self.select_object
        for topic in (Topic.MODEL, Topic.DOCUMENT, Topic.PREDICTION, Topic.ACTIVITY):
            self.state.bus.subscribe(topic, self.update_actions)
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.update_actions()

    def _bind_shortcuts(self) -> None:
        def key(sequence: str, action, allow_in_table: bool = True) -> None:
            def handler(event):
                if not allow_in_table and isinstance(event.widget, (ttk.Treeview, ttk.Combobox)):
                    return None
                action()
                return "break"
            self.root.bind(sequence, handler)

        for sequence in ("<Control-o>", "<Control-O>"):
            key(sequence, self.open_image_dialog)
        for sequence in ("<F5>", "<Control-Return>"):
            key(sequence, self.predict)
        key("<Delete>", self.clear)
        for sequence in ("<Prior>", "<Alt-Left>"):
            key(sequence, self.previous_image, allow_in_table=sequence != "<Prior>")
        for sequence in ("<Next>", "<Alt-Right>"):
            key(sequence, self.next_image, allow_in_table=sequence != "<Next>")
        for sequence in ("<Control-s>", "<Control-S>"):
            key(sequence, self.save_overlay_dialog)
        for sequence in ("<Control-e>", "<Control-E>"):
            key(sequence, self.export_results_dialog)
        for sequence in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
            key(sequence, self.zoom_in)
        for sequence in ("<Control-minus>", "<Control-KP_Subtract>"):
            key(sequence, self.zoom_out)
        key("<Control-Key-0>", self.fit)
        key("<Control-Key-1>", self.actual_size)
        key("<Control-h>", self.toggle_segments)
        key("<Control-t>", self.toggle_outlines)
        key("<Escape>", lambda: self.select_object(None))
        key("<F1>", self.show_shortcuts)

    # ---- model -------------------------------------------------------------------
    def start(self) -> None:
        self.state.set_activity(message="Loading the V7 model… (about 15 s)")
        self.worker.submit(self.engine.load, on_success=self._model_loaded,
                           on_error=self._model_failed)

    def _model_loaded(self, info) -> None:
        self.state.model_ready(info, self.engine.classes)
        hint = " Press Predict." if self.state.document else " Insert an image to start."
        self.state.set_activity(
            message=f"Model ready on {info.device} (loaded in {info.load_seconds:.0f} s).{hint}")
        if self.window.auto_predict.get() and self.state.document and not self.state.prediction:
            self.predict()

    def _model_failed(self, error: Exception) -> None:
        self.state.model_failed(str(error))
        self.state.set_activity(message="The model could not be loaded. See Help → Open the log folder.")
        dialogs.show_error(self.root, "V7 model", "The V7 model could not be loaded.", str(error))

    # ---- photos --------------------------------------------------------------------
    def open_image_dialog(self) -> None:
        if self.state.predicting:
            return
        folder = Path(self.settings.last_folder) if self.settings.last_folder else Path.home() / "Desktop"
        chosen = filedialog.askopenfilename(
            parent=self.root, title="Insert image", filetypes=FILE_DIALOG_TYPES,
            initialdir=str(folder if folder.is_dir() else Path.home()))
        if chosen:
            self.open_image(Path(chosen))

    def open_image(self, path: Path) -> bool:
        if self.state.predicting:
            return False
        try:
            raw = read_image(path)
        except Exception as error:
            log.exception("could not open %s", path)
            if not path.exists():
                self.settings.remove_recent(path)
                self.window.refresh_recent(self.settings.recent_files)
            dialogs.show_error(self.root, "Insert image", "This image could not be opened.",
                               f"{path}\n\n{error}")
            return False
        self._generation += 1
        self.navigator.set_current(path)
        self.state.set_document(ImageDocument(path, raw, to_display_uint8(raw)),
                                self.navigator.position)
        self.settings.add_recent(path)
        self.settings.last_folder = str(path.parent)
        self.window.refresh_recent(self.settings.recent_files)
        ready = self.state.is_model_ready
        self.state.set_activity(message=f"Opened {path.name}."
                                + (" Press Predict." if ready else " The model is still loading…"))
        if ready and self.window.auto_predict.get():
            self.predict()
        return True

    def previous_image(self) -> None:
        self._step(-1)

    def next_image(self) -> None:
        self._step(+1)

    def _step(self, step: int) -> None:
        target = self.navigator.peek(step) if self.state.document else None
        if target is not None and not self.state.predicting:
            self.open_image(target)

    def clear_recent(self) -> None:
        self.settings.recent_files = []
        self.window.refresh_recent([])

    # ---- prediction ------------------------------------------------------------------
    def predict(self) -> None:
        state = self.state
        if not (state.is_model_ready and state.document is not None) or state.predicting:
            return
        document, generation = state.document, self._generation
        output_dir = self.workspace.new_prediction_dir(document.path.stem)
        state.set_activity(predicting=True, message=f"Predicting {document.name}…")
        self.worker.submit(self.engine.predict, document.path, output_dir,
                           on_success=lambda result: self._predicted(generation, result),
                           on_error=lambda error: self._prediction_failed(generation, error))

    def _predicted(self, generation: int, prediction: Prediction) -> None:
        self.state.set_activity(predicting=False)
        document = self.state.document
        if generation != self._generation or document is None:
            return                                   # the photo changed meanwhile
        # Show exactly the pixels the model predicted on (matters for 16-bit photos).
        document.display_rgb = self.engine.model_rgb(document.raw)
        height, width = document.display_rgb.shape[:2]
        if prediction.instance_map.shape[:2] != (height, width):
            log.warning("instance map %s does not match the photo %s; resizing",
                        prediction.instance_map.shape, (height, width))
            prediction.instance_map = resize_nearest(prediction.instance_map, width, height)
        self.state.set_prediction(prediction)
        count = len(prediction.objects)
        self.state.set_activity(message=f"Predicted {count} object{'s' if count != 1 else ''} "
                                f"in {prediction.wall_seconds:.1f} s.")

    def _prediction_failed(self, generation: int, error: Exception) -> None:
        self.state.set_activity(predicting=False, message="The prediction failed. See the log.")
        if generation == self._generation:
            dialogs.show_error(self.root, "Predict", "The prediction failed.", str(error))

    def clear(self) -> None:
        if self.state.predicting or self.state.document is None:
            return
        self._generation += 1
        self.navigator.clear()
        self.state.set_document(None)
        self.state.set_activity(message="Cleared. Insert another image.")

    # ---- saving ----------------------------------------------------------------------
    def save_overlay_dialog(self) -> None:
        state = self.state
        if state.prediction is None or state.predicting:
            return
        chosen = filedialog.asksaveasfilename(
            parent=self.root, title="Save overlay image", defaultextension=".png",
            initialfile=f"{state.document.path.stem}_v7_overlay.png",
            initialdir=self.settings.last_folder or str(Path.home()),
            filetypes=(("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("Bitmap", "*.bmp")))
        if chosen:
            self.save_overlay_to(Path(chosen))

    def save_overlay_to(self, path: Path) -> Path | None:
        state = self.state
        try:
            export.save_overlay(path, state.document, state.prediction, state.display, state.classes)
        except Exception as error:
            log.exception("saving the overlay failed")
            dialogs.show_error(self.root, "Save overlay", "The image could not be saved.", str(error))
            return None
        state.set_activity(message=f"Saved {path}")
        return path

    def export_results_dialog(self) -> None:
        state = self.state
        if state.prediction is None or state.predicting:
            return
        folder = filedialog.askdirectory(parent=self.root, title="Export results to folder",
                                         initialdir=self.settings.last_folder or str(Path.home()))
        if not folder:
            return
        target = self.export_results_to(Path(folder))
        if target is not None and messagebox.askyesno(
                "Export", f"Results exported to\n{target}\n\nOpen the folder?", parent=self.root):
            os.startfile(target)

    def export_results_to(self, folder: Path) -> Path | None:
        state = self.state
        try:
            target = export.export_results(folder, state.document, state.prediction,
                                           state.display, state.classes, state.model_info)
        except Exception as error:
            log.exception("export failed")
            dialogs.show_error(self.root, "Export", "The results could not be exported.", str(error))
            return None
        state.set_activity(message=f"Exported to {target}")
        return target

    # ---- view --------------------------------------------------------------------------
    def zoom_in(self) -> None:
        self.window.image_view.zoom_in()

    def zoom_out(self) -> None:
        self.window.image_view.zoom_out()

    def fit(self) -> None:
        self.window.image_view.fit()

    def actual_size(self) -> None:
        self.window.image_view.actual_size()

    def toggle_segments(self) -> None:
        self.update_display(show_segments=not self.state.display.show_segments)

    def toggle_outlines(self) -> None:
        self.update_display(outlines=not self.state.display.outlines)

    def set_layer(self, layer: str) -> None:
        if layer in ARTIFACT_LAYERS and self.state.prediction is None:
            self.window.layer.set(self.state.display.layer)   # not available yet
            return
        self.update_display(layer=layer)

    def update_display(self, **changes) -> None:
        if changes.get("layer") in ARTIFACT_LAYERS and self.state.prediction is None:
            changes.pop("layer")
        if changes:
            self.state.update_display(**changes)

    def set_class_visible(self, class_id: int, visible: bool) -> None:
        hidden = set(self.state.display.hidden_classes)
        (hidden.discard if visible else hidden.add)(class_id)
        self.update_display(hidden_classes=frozenset(hidden))

    def select_object(self, object_id: int | None) -> None:
        self.state.select(object_id)
        selected = self.state.selected_object
        if selected is not None:
            self.window.image_view.ensure_visible(selected)

    def auto_predict_changed(self) -> None:
        if self.window.auto_predict.get() and self.state.document and not self.state.prediction:
            self.predict()

    # ---- enabling ----------------------------------------------------------------------
    def update_actions(self) -> None:
        if self.window is None:
            return
        state = self.state
        idle = not state.predicting
        has_document = state.document is not None
        has_prediction = state.prediction is not None
        enabled = {
            "open": idle,
            "previous": idle and has_document and self.navigator.has_previous,
            "next": idle and has_document and self.navigator.has_next,
            "predict": idle and has_document and state.is_model_ready,
            "clear": idle and has_document,
            "save": idle and has_prediction,
            "export": idle and has_prediction,
            "zoom": has_document,
            "layers": has_prediction,
        }
        for action, on in enabled.items():
            self.window.set_enabled(action, on)

    # ---- help and exit ---------------------------------------------------------------------
    def show_shortcuts(self) -> None:
        dialogs.show_shortcuts(self.root, self.scale)

    def show_about(self) -> None:
        dialogs.show_about(self.root, self.scale, self.state.model_info, self.state.model_error)

    def open_log_folder(self) -> None:
        os.startfile(self.paths.log_dir)

    def quit(self) -> None:
        settings, display = self.settings, self.state.display
        try:
            settings.window_maximized = self.root.state() == "zoomed"
            if not settings.window_maximized:
                settings.window_geometry = self.root.winfo_geometry()
            settings.side_panel_width = self.window.side_panel_width()
            settings.opacity = display.opacity
            settings.show_segments = display.show_segments
            settings.show_outlines = display.outlines
            settings.auto_predict = bool(self.window.auto_predict.get())
            settings.save(self.paths.settings_file)
        except Exception:
            log.exception("could not save settings")
        self.workspace.close()
        log.info("closed")
        self.root.destroy()
