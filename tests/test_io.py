import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from v7_segmenter.imaging.io import is_image_file, read_image, to_display_uint8, write_image


class ImageIoTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp()) / "tëst ünïcode folder"
        self.folder.mkdir()

    def test_round_trip_through_a_non_ascii_path(self):
        rgb = np.random.default_rng(0).integers(0, 256, (7, 9, 3), dtype=np.uint8)
        path = self.folder / "image 1.png"
        write_image(path, rgb)
        self.assertTrue(np.array_equal(read_image(path), rgb))

    def test_grayscale_and_alpha_become_rgb(self):
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        cv2.imencode(".png", gray)[1].tofile(str(self.folder / "gray.png"))
        rgba = np.zeros((3, 4, 4), dtype=np.uint8)
        rgba[..., 2] = 200                      # red in BGRA
        cv2.imencode(".png", rgba)[1].tofile(str(self.folder / "alpha.png"))
        gray_read = read_image(self.folder / "gray.png")
        self.assertEqual(gray_read.shape, (3, 4, 3))
        self.assertTrue(np.array_equal(gray_read[..., 0], gray))
        self.assertEqual(read_image(self.folder / "alpha.png")[0, 0].tolist(), [200, 0, 0])

    def test_unreadable_file_raises_value_error(self):
        path = self.folder / "broken.png"
        path.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            read_image(path)

    def test_display_conversion(self):
        self.assertEqual(to_display_uint8(np.array([[[65535, 0, 256]]], dtype=np.uint16)).tolist(),
                         [[[255, 0, 1]]])
        self.assertEqual(to_display_uint8(np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)).tolist(),
                         [[[0, 127, 255]]])

    def test_image_extensions(self):
        self.assertTrue(is_image_file(Path("a.PNG")))
        self.assertTrue(is_image_file(Path("b.tiff")))
        self.assertFalse(is_image_file(Path("c.txt")))


if __name__ == "__main__":
    unittest.main()
