"""Loads the real model and predicts one photo (about 20 s, so opt-in):

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


if __name__ == "__main__":
    unittest.main()
