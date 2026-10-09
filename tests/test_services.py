import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from v7_segmenter.domain import DetectedObject, ImageDocument, Prediction
from v7_segmenter.events import EventBus
from v7_segmenter.services.navigation import FolderNavigator
from v7_segmenter.services.state import AppState, Topic
from v7_segmenter.settings import Settings


class SettingsTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "settings.json"

    def test_round_trip(self):
        settings = Settings(opacity=0.3, auto_predict=True, recent_files=["C:/a.png"])
        settings.save(self.path)
        self.assertEqual(Settings.load(self.path), settings)

    def test_missing_or_corrupt_file_gives_defaults(self):
        self.assertEqual(Settings.load(self.path), Settings())
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(Settings.load(self.path), Settings())

    def test_invalid_values_are_ignored(self):
        self.path.write_text(json.dumps({"opacity": "x", "auto_predict": 1, "recent_files": [1],
                                         "show_outlines": False, "unknown": 5}), encoding="utf-8")
        loaded = Settings.load(self.path)
        self.assertEqual(loaded.opacity, Settings().opacity)
        self.assertFalse(loaded.auto_predict)
        self.assertEqual(loaded.recent_files, [])
        self.assertFalse(loaded.show_outlines)

    def test_recent_files_are_unique_newest_first_and_capped(self):
        settings = Settings()
        for number in range(12):
            settings.add_recent(f"C:/img{number}.png")
        settings.add_recent("c:/IMG5.png")          # same file, different case
        self.assertEqual(settings.recent_files[0], "c:/IMG5.png")
        self.assertEqual(len(settings.recent_files), Settings.MAX_RECENT)
        self.assertEqual(sum(p.lower() == "c:/img5.png" for p in settings.recent_files), 1)


class NavigatorTest(unittest.TestCase):
    def test_natural_order_and_stepping(self):
        folder = Path(tempfile.mkdtemp())
        for name in ("img10.png", "img2.png", "img1.png", "notes.txt"):
            (folder / name).write_bytes(b"x")
        navigator = FolderNavigator()
        navigator.set_current(folder / "img2.png")
        self.assertEqual(navigator.position, (2, 3))
        self.assertEqual(navigator.peek(-1).name, "img1.png")
        self.assertEqual(navigator.peek(+1).name, "img10.png")
        self.assertIsNone(navigator.peek(+2))
        navigator.clear()
        self.assertIsNone(navigator.position)
        self.assertFalse(navigator.has_next)


class StateTest(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.published = []
        for topic in vars(Topic).values():
            if isinstance(topic, str) and not topic.startswith("_"):
                self.bus.subscribe(topic, lambda t=topic: self.published.append(t))
        self.state = AppState(self.bus)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        self.document = ImageDocument(Path("a.png"), image, image)
        objects = [DetectedObject(1, 1, "Rectangle", 0.9, 4, (0, 0, 1, 1), (0.5, 0.5)),
                   DetectedObject(2, 3, "circle", 0.2, 4, (5, 5, 6, 6), (5.5, 5.5))]
        self.prediction = Prediction(Path("a.png"), np.zeros((10, 10), np.uint16), objects,
                                     100.0, 20.0, Path("."))

    def test_new_document_clears_the_prediction_and_selection(self):
        self.state.set_document(self.document)
        self.state.set_prediction(self.prediction)
        self.state.select(1)
        self.state.set_document(None)
        self.assertIsNone(self.state.prediction)
        self.assertIsNone(self.state.selected_id)
        self.assertIn(Topic.DOCUMENT, self.published)

    def test_filters_hide_objects_and_drop_a_hidden_selection(self):
        self.state.set_document(self.document)
        self.state.set_prediction(self.prediction)
        self.state.select(2)
        self.state.update_display(min_confidence=0.5)
        self.assertEqual([o.id for o in self.state.visible_objects()], [1])
        self.assertIsNone(self.state.selected_id)
        self.state.update_display(min_confidence=0.0, hidden_classes=frozenset({1}))
        self.assertEqual([o.id for o in self.state.visible_objects()], [2])

    def test_selecting_the_same_object_twice_publishes_once(self):
        self.state.set_prediction(self.prediction)
        self.published.clear()
        self.state.select(1)
        self.state.select(1)
        self.assertEqual(self.published.count(Topic.SELECTION), 1)


if __name__ == "__main__":
    unittest.main()
