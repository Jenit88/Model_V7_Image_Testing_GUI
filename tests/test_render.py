import unittest

import numpy as np

from v7_segmenter.domain import DEFAULT_CLASSES, DetectedObject
from v7_segmenter.imaging.render import (
    HIGHLIGHT_RGB, OverlayStyle, class_lookup, colour_table, compose_overlay,
    full_resolution_overlay, resize_nearest,
)


def item(object_id: int, class_id: int, confidence: float) -> DetectedObject:
    return DetectedObject(object_id, class_id, "test", confidence, 36, (0, 0, 5, 5), (2.5, 2.5))


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((20, 20, 3), 100, dtype=np.uint8)
        self.instances = np.zeros((20, 20), dtype=np.uint16)
        self.instances[2:8, 2:8] = 1
        self.instances[10:16, 10:16] = 2
        self.objects = [item(1, 1, 0.9), item(2, 4, 0.3)]
        self.colours = colour_table(DEFAULT_CLASSES)

    def lookup(self, hidden=(), min_confidence=0.0):
        return class_lookup(self.objects, set(hidden), min_confidence, 3)

    def test_lookup_applies_class_and_confidence_filters(self):
        self.assertEqual(self.lookup().tolist(), [0, 1, 4])
        self.assertEqual(self.lookup(hidden={1}).tolist(), [0, 0, 4])
        self.assertEqual(self.lookup(min_confidence=0.5).tolist(), [0, 1, 0])

    def test_objects_are_tinted_and_background_is_untouched(self):
        out = compose_overlay(self.rgb, self.instances, self.lookup(), self.colours,
                              OverlayStyle(opacity=0.5, outlines=False))
        self.assertEqual(out[0, 0].tolist(), [100, 100, 100])
        expected = (100 * 0.5 + self.colours[1].astype(float) * 0.5).astype(np.uint8)
        self.assertEqual(out[4, 4].tolist(), expected.tolist())

    def test_outline_is_drawn_inside_the_object_in_its_class_colour(self):
        out = compose_overlay(self.rgb, self.instances, self.lookup(), self.colours,
                              OverlayStyle(opacity=0.0, outlines=True, outline_px=1))
        self.assertEqual(out[2, 2].tolist(), self.colours[1].tolist())
        self.assertEqual(out[1, 2].tolist(), [100, 100, 100])     # outside: untouched
        self.assertEqual(out[4, 4].tolist(), [100, 100, 100])     # inside: no tint at 0%

    def test_hidden_objects_are_not_drawn(self):
        out = compose_overlay(self.rgb, self.instances, self.lookup(hidden={1}), self.colours,
                              OverlayStyle(opacity=0.5))
        self.assertTrue(np.array_equal(out[2:8, 2:8], self.rgb[2:8, 2:8]))

    def test_selected_object_gets_a_white_ring(self):
        out = compose_overlay(self.rgb, self.instances, self.lookup(), self.colours,
                              OverlayStyle(opacity=0.5), highlight_id=1)
        white = np.all(out == HIGHLIGHT_RGB, axis=-1)
        self.assertTrue(white[1:9, 1:9].any())
        # The ring is thickened, so it reaches a few pixels past object 1 --
        # but never into the far object.
        self.assertFalse(white[12:, 12:].any())

    def test_input_is_not_modified(self):
        before = self.rgb.copy()
        compose_overlay(self.rgb, self.instances, self.lookup(), self.colours, OverlayStyle())
        self.assertTrue(np.array_equal(self.rgb, before))

    def test_full_resolution_overlay_keeps_the_size(self):
        out = full_resolution_overlay(self.rgb, self.instances, self.lookup(), self.colours,
                                      OverlayStyle())
        self.assertEqual(out.shape, self.rgb.shape)

    def test_nearest_resize_keeps_labels_and_dtype(self):
        for dtype in (np.uint16, np.int32):
            resized = resize_nearest(self.instances.astype(dtype), 40, 30)
            self.assertEqual(resized.shape, (30, 40))
            self.assertEqual(resized.dtype, dtype)
            self.assertEqual(set(np.unique(resized).tolist()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
