"""Right-hand panel: prediction summary, display options and the object table."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from v7_segmenter.services.state import LAYERS, AppState, ModelStatus, Topic
from v7_segmenter.ui import theme
from v7_segmenter.ui.theme import px

BASIC_LAYERS = LAYERS[:2]
COLUMNS = (
    ("id", "#", 44, "e"),
    ("class", "Class", 130, "w"),
    ("confidence", "Confidence", 84, "e"),
    ("area", "Area (px)", 84, "e"),
    ("centre", "Centre (x, y)", 104, "e"),
)


def _darker(hex_colour: str, factor: float = 0.75) -> str:
    rgb = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(v * factor):02x}" for v in rgb)


class SidePanel(ttk.Frame):
    def __init__(self, master: tk.Misc, state: AppState, scale: float):
        super().__init__(master, padding=(px(10, scale), px(8, scale), px(10, scale), px(4, scale)))
        self.state = state
        self.scale = scale
        self._updating = False
        self._sort = ("id", False)

        # Set by the controller.
        self.on_class_toggle: Callable[[int, bool], None] = lambda class_id, visible: None
        self.on_display: Callable[..., None] = lambda **changes: None
        self.on_select: Callable[[int | None], None] = lambda object_id: None

        self._build_summary()
        self._build_display()
        self._build_objects()

        bus = state.bus
        bus.subscribe(Topic.MODEL, self._refresh_classes)
        for topic in (Topic.DOCUMENT, Topic.PREDICTION, Topic.DISPLAY):
            bus.subscribe(topic, self._refresh_all)
        bus.subscribe(Topic.SELECTION, self._refresh_selection)
        self._refresh_classes()

    # ---- construction ----------------------------------------------------------
    def _section(self, title: str) -> ttk.Labelframe:
        frame = ttk.Labelframe(self, text=title, style="Section.TLabelframe",
                               padding=(px(10, self.scale), px(6, self.scale)))
        frame.pack(fill="x", pady=(0, px(10, self.scale)))
        return frame

    def _build_summary(self) -> None:
        section = self._section("Prediction")
        self.summary_var = tk.StringVar(value="No image inserted.")
        ttk.Label(section, textvariable=self.summary_var, style="Info.TLabel",
                  wraplength=px(300, self.scale), justify="left").pack(anchor="w")
        self.class_frame = ttk.Frame(section)
        self.class_frame.pack(fill="x", pady=(px(6, self.scale), 0))
        self.class_rows: dict[int, tuple[tk.BooleanVar, ttk.Label]] = {}

    def _build_display(self) -> None:
        s = self.scale
        section = self._section("Display")
        section.columnconfigure(1, weight=1)

        self.show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(section, text="Show segments  (Ctrl+H)", variable=self.show_var,
                        command=lambda: self._emit(show_segments=self.show_var.get())
                        ).grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(section, text="View").grid(row=1, column=0, sticky="w", pady=(px(6, s), 0))
        self.layer_var = tk.StringVar()
        self.layer_box = ttk.Combobox(section, textvariable=self.layer_var, state="readonly")
        self.layer_box.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(px(8, s), 0),
                            pady=(px(6, s), 0))
        self.layer_box.bind("<<ComboboxSelected>>", self._on_layer)

        ttk.Label(section, text="Opacity").grid(row=2, column=0, sticky="w", pady=(px(6, s), 0))
        self.opacity_var = tk.DoubleVar(value=45)
        ttk.Scale(section, from_=0, to=100, variable=self.opacity_var,
                  command=lambda value: self._emit(opacity=float(value) / 100.0)
                  ).grid(row=2, column=1, sticky="ew", padx=(px(8, s), 0), pady=(px(6, s), 0))
        self.opacity_text = ttk.Label(section, width=5, anchor="e")
        self.opacity_text.grid(row=2, column=2, sticky="e", pady=(px(6, s), 0))

        self.outline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(section, text="Outlines  (Ctrl+T)", variable=self.outline_var,
                        command=lambda: self._emit(outlines=self.outline_var.get())
                        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(px(6, s), 0))

        ttk.Label(section, text="Hide objects below confidence").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(px(8, s), 0))
        self.confidence_var = tk.DoubleVar(value=0)
        ttk.Scale(section, from_=0, to=100, variable=self.confidence_var,
                  command=lambda value: self._emit(min_confidence=round(float(value)) / 100.0)
                  ).grid(row=5, column=0, columnspan=2, sticky="ew")
        self.confidence_text = ttk.Label(section, width=5, anchor="e")
        self.confidence_text.grid(row=5, column=2, sticky="e")
        ttk.Label(section, text="Only hides objects on screen; the model output is unchanged.",
                  style="Muted.TLabel", wraplength=px(300, s), justify="left"
                  ).grid(row=6, column=0, columnspan=3, sticky="w")

    def _build_objects(self) -> None:
        s = self.scale
        section = ttk.Labelframe(self, text="Objects", style="Section.TLabelframe",
                                 padding=(px(6, s), px(6, s)))
        section.pack(fill="both", expand=True)
        self.objects_var = tk.StringVar(value="")
        ttk.Label(section, textvariable=self.objects_var, style="Muted.TLabel").pack(anchor="w")
        holder = ttk.Frame(section)
        holder.pack(fill="both", expand=True, pady=(px(4, s), 0))
        self.table = ttk.Treeview(holder, columns=[c[0] for c in COLUMNS], show="headings",
                                  selectmode="browse")
        for key, title, width, anchor in COLUMNS:
            self.table.heading(key, text=title, command=lambda k=key: self._sort_by(k))
            self.table.column(key, width=px(width, s), minwidth=px(36, s), anchor=anchor,
                              stretch=key == "class")
        scroll = ttk.Scrollbar(holder, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", self._on_table_select)

    # ---- events ------------------------------------------------------------------
    def _emit(self, **changes) -> None:
        if not self._updating:
            self.on_display(**changes)

    def _on_layer(self, event=None) -> None:
        label = self.layer_var.get()
        key = next((k for k, text, _ in LAYERS if text == label), "overlay")
        self._emit(layer=key)

    def _on_class(self, class_id: int) -> None:
        if not self._updating:
            self.on_class_toggle(class_id, self.class_rows[class_id][0].get())

    def _on_table_select(self, event=None) -> None:
        if self._updating:
            return
        selection = self.table.selection()
        self.on_select(int(selection[0]) if selection else None)

    def _sort_by(self, key: str) -> None:
        column, reverse = self._sort
        self._sort = (key, not reverse if column == key else False)
        self._refresh_table()

    # ---- refresh from state ------------------------------------------------------
    def _refresh_classes(self) -> None:
        """(Re)build the class rows; names and colours come from model_v7 once loaded."""
        for child in self.class_frame.winfo_children():
            child.destroy()
        self.class_rows.clear()
        for item in self.state.classes:
            row = ttk.Frame(self.class_frame)
            row.pack(fill="x", pady=1)
            tk.Label(row, bg=item.hex, width=2, relief="flat").pack(side="left", padx=(0, 6))
            variable = tk.BooleanVar(value=item.id not in self.state.display.hidden_classes)
            ttk.Checkbutton(row, text=item.name, variable=variable,
                            command=lambda cid=item.id: self._on_class(cid)).pack(side="left")
            count = ttk.Label(row, text="", anchor="e")
            count.pack(side="right")
            self.class_rows[item.id] = (variable, count)
            self.table.tag_configure(f"class{item.id}", foreground=_darker(item.hex))
        self._refresh_all()

    def _refresh_all(self) -> None:
        self._updating = True
        try:
            self._refresh_summary()
            self._refresh_display()
            self._refresh_table()
        finally:
            self._updating = False
        self._refresh_selection()

    def _refresh_summary(self) -> None:
        state, prediction = self.state, self.state.prediction
        if state.document is None:
            text = "No image inserted."
        elif prediction is None:
            text = ("Press Predict to segment this image." if state.model_status is ModelStatus.READY
                    else "The model is still loading…" if state.model_status is ModelStatus.LOADING
                    else "The model is not available.")
        else:
            total = len(prediction.objects)
            text = (f"{total} object{'s' if total != 1 else ''} found in "
                    f"{prediction.wall_seconds:.1f} s (model {prediction.model_ms / 1000:.1f} s).")
        self.summary_var.set(text)
        counts = prediction.counts_by_class() if prediction else {}
        shown = {}
        for item in state.visible_objects():
            shown[item.class_id] = shown.get(item.class_id, 0) + 1
        for class_id, (variable, label) in self.class_rows.items():
            variable.set(class_id not in state.display.hidden_classes)
            if prediction is None:
                label.configure(text="")
            else:
                total, visible = counts.get(class_id, 0), shown.get(class_id, 0)
                label.configure(text=f"{total}" if visible == total or class_id in
                                state.display.hidden_classes else f"{visible} of {total}")

    def _refresh_display(self) -> None:
        display, has_prediction = self.state.display, self.state.prediction is not None
        self.show_var.set(display.show_segments)
        self.outline_var.set(display.outlines)
        self.opacity_var.set(display.opacity * 100)
        self.opacity_text.configure(text=f"{display.opacity * 100:.0f}%")
        self.confidence_var.set(display.min_confidence * 100)
        self.confidence_text.configure(text=f"{display.min_confidence:.2f}")
        layers = LAYERS if has_prediction else BASIC_LAYERS
        self.layer_box.configure(values=[text for _, text, _ in layers])
        self.layer_var.set(next((text for key, text, _ in LAYERS if key == display.layer),
                                LAYERS[0][1]))

    def _refresh_table(self) -> None:
        prediction = self.state.prediction
        visible = self.state.visible_objects()
        # Rebuild only when the rows would change, so moving the opacity slider
        # does not scroll the table back to the top.
        key = (id(prediction), tuple(item.id for item in visible), self._sort)
        if key == getattr(self, "_table_key", None):
            return
        self._table_key = key
        was = self._updating
        self._updating = True
        try:
            self.table.delete(*self.table.get_children())
            if prediction is None:
                self.objects_var.set("")
                return
            column, reverse = self._sort
            keys = {"id": lambda o: o.id, "class": lambda o: (o.class_name, o.id),
                    "confidence": lambda o: o.confidence, "area": lambda o: o.area_px,
                    "centre": lambda o: (o.centroid_xy[1], o.centroid_xy[0])}
            for item in sorted(visible, key=keys[column], reverse=reverse):
                self.table.insert("", "end", iid=str(item.id), tags=(f"class{item.class_id}",),
                                  values=(item.id, item.class_name, f"{item.confidence:.2f}",
                                          f"{item.area_px:,}", item.centre_text))
            hidden = len(prediction.objects) - len(visible)
            self.objects_var.set(f"{len(visible)} shown" + (f", {hidden} hidden by the filters"
                                                           if hidden else "")
                                 + ".  Click a row or an object to highlight it.")
        finally:
            self._updating = was

    def _refresh_selection(self) -> None:
        self._updating = True
        try:
            selected = self.state.selected_id
            if selected is not None and self.table.exists(str(selected)):
                self.table.selection_set(str(selected))
                self.table.see(str(selected))
            else:
                self.table.selection_remove(*self.table.selection())
        finally:
            self._updating = False
