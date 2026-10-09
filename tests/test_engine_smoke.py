"""Loads the real model and predicts one photo in both modes (about 25 s, so opt-in):

    set V7_MODEL_TESTS=1
    set V7_TEST_IMAGE=C:\\path\\to\\a\\test\\photo.png
    python -m unittest tests.test_engine_smoke -v
"""
import os
import tempfile
import unittest
from pathlib import Path

from v7_segmenter.imaging.io import read_image
from v7_segmenter.inference.engine import V7Engine
from v7_segmenter.paths import default_paths


@unittest.skipUnless(os.environ.get("V7_MODEL_TESTS") == "1", "set V7_MODEL_TESTS=1 to run")
class EngineSmokeTest(unittest.TestCase):
    def test_load_and_predict(self):
        image = Path(os.environ["V7_TEST_IMAGE"])
        engine = V7Engine(default_paths())
        info = engine.load()
        self.assertTrue(engine.ready)
        self.assertEqual(len(engine.classes), 4)
        self.assertTrue(info.sha256_prefix)
        prediction = engine.predict(image, Path(tempfile.mkdtemp()))
        height, width = read_image(image).shape[:2]
        self.assertEqual(prediction.instance_map.shape[:2], (height, width))
        self.assertGreater(len(prediction.objects), 0)
        for item in prediction.objects:
            self.assertTrue((prediction.instance_map == item.id).any())
        for path in prediction.artifacts.values():
            self.assertTrue(path.is_file(), path)

    def test_fast_mode_matches_exact_mode(self):
        """The compiled runner may round differently, so a few edge pixels can
        change; the objects found must stay the same."""
        image = Path(os.environ["V7_TEST_IMAGE"])
        engine = V7Engine(default_paths())
        engine.load()
        engine.exact_mode = True
        exact = engine.predict(image, Path(tempfile.mkdtemp()))
        engine.exact_mode = False
        engine.compile_fast()
        self.assertTrue(engine.fast_ready)
        fast = engine.predict(image, Path(tempfile.mkdtemp()))
        self.assertEqual(sorted(o.class_id for o in fast.objects),
                         sorted(o.class_id for o in exact.objects))
        same_object = (fast.instance_map > 0) == (exact.instance_map > 0)
        self.assertGreater(same_object.mean(), 0.999)


if __name__ == "__main__":
    unittest.main()
