"""The four PCB classes the V7 model segments."""
from __future__ import annotations

from dataclasses import dataclass

SHORT_NAMES = {"Rectangle": "R", "Rectangle_concave": "Rc", "circle": "c", "circle_full": "cf"}


@dataclass(frozen=True)
class SegmentClass:
    id: int
    name: str
    rgb: tuple[int, int, int]

    @property
    def short(self) -> str:
        return SHORT_NAMES.get(self.name, self.name[:2])

    @property
    def hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)


# Mirrors model_v7.CLASS_NAMES / CLASS_COLOURS_RGB. Used until the model code
# is loaded, after which classes_from_model() replaces it with the real thing.
DEFAULT_CLASSES: tuple[SegmentClass, ...] = (
    SegmentClass(1, "Rectangle", (230, 65, 65)),
    SegmentClass(2, "Rectangle_concave", (255, 165, 45)),
    SegmentClass(3, "circle", (60, 180, 90)),
    SegmentClass(4, "circle_full", (65, 135, 230)),
)


def classes_from_model(class_names: dict, colours) -> tuple[SegmentClass, ...]:
    """Build the class list from model_v7's own constants (background excluded)."""
    return tuple(
        SegmentClass(int(class_id), str(name), tuple(int(v) for v in colours[int(class_id)]))
        for class_id, name in sorted(class_names.items())
        if int(class_id) > 0
    )
