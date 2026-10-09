"""Scripted end-to-end check of the running application.

Drives the real window through the controller -- open, predict, select, zoom,
switch layers, filter, export, clear -- taking a screenshot after each step,
then closes. Settings are not saved, so a check leaves no trace.

    python run_gui.py --demo IMAGE --screenshots C:/temp/check
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import ImageGrab

if TYPE_CHECKING:
    from v7_segmenter.ui.controller import MainController
    from v7_segmenter.ui.main_window import MainWindow

log = logging.getLogger(__name__)


def run_demo(controller: "MainController", window: "MainWindow", image: Path,
             screenshots: Path | None) -> None:
    root, state = controller.root, controller.state
    report: list[str] = []
    controller.settings.save = lambda path: None          # leave the user's settings alone

    def grab(name: str) -> None:
        if screenshots is None:
            return
        root.update()
        x, y = root.winfo_rootx(), root.winfo_rooty()
        target = screenshots.parent / f"{screenshots.name}_{name}.png"
        ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height())).save(target)

    def wait(condition, then, every: int = 300) -> None:
        if condition():
            root.after(600, then)
        else:
            root.after(every, wait, condition, then, every)

    def step_start() -> None:
        grab("1_start")
        wait(lambda: state.is_model_ready, step_open)

    def step_open() -> None:
        controller.open_image(image)
        root.after(700, step_predict)

    def step_predict() -> None:
        grab("2_opened")
        controller.predict()
        wait(lambda: state.prediction is not None and not state.predicting, step_predicted)

    def step_predicted() -> None:
        grab("3_predicted")
        objects = state.prediction.objects
        report.append(f"{len(objects)} objects in {state.prediction.wall_seconds:.1f} s")
        largest = max(objects, key=lambda o: o.area_px)
        controller.select_object(largest.id)
        controller.zoom_in()
        controller.zoom_in()
        window.image_view.ensure_visible(largest)
        root.after(700, step_layers)

    def step_layers() -> None:
        grab("4_selected_zoomed")
        controller.fit()
        controller.set_layer("semantic")
        root.after(900, step_filtered)

    def step_filtered() -> None:
        grab("5_semantic_layer")
        controller.set_layer("overlay")
        controller.set_class_visible(state.classes[0].id, False)
        controller.update_display(min_confidence=0.5)
        root.after(700, step_export)

    def step_export() -> None:
        grab("6_filtered")
        folder = Path(tempfile.mkdtemp(prefix="v7_demo_export_"))
        target = controller.export_results_to(folder)
        saved = controller.save_overlay_to(folder / "overlay_saved.png")
        files = sorted(p.name for p in target.iterdir()) if target else []
        report.append(f"export -> {target} ({len(files)} files: {', '.join(files)})")
        report.append(f"overlay saved -> {saved}")
        controller.update_display(min_confidence=0.0)
        controller.set_class_visible(state.classes[0].id, True)
        controller.clear()
        root.after(700, step_done)

    def step_done() -> None:
        grab("7_cleared")
        report.append(f"after clear: document={state.document}, message={state.message!r}")
        for line in report:
            print("DEMO", line, flush=True)
            log.info("demo: %s", line)
        controller.quit()

    root.after(1200, step_start)
