import os
import tempfile
import unittest
from pathlib import Path

from v7_segmenter.inference.engine import ModelLoadError, V7Engine, is_lfs_pointer
from v7_segmenter.paths import AppPaths

POINTER = (b"version https://git-lfs.github.com/spec/v1\n"
           b"oid sha256:872f9bc06ecf0a2e0000000000000000000000000000000000000000000000\n"
           b"size 231039834\n")


class ModelFileTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.paths = AppPaths(self.root)
        self.paths.resource_dir.mkdir()
        self.paths.model_code.write_text("# stand-in\n", encoding="utf-8")

    def test_recognises_a_git_lfs_placeholder(self):
        self.paths.model_file.write_bytes(POINTER)
        self.assertTrue(is_lfs_pointer(self.paths.model_file))

    def test_a_real_file_is_not_a_placeholder(self):
        self.paths.model_file.write_bytes(os.urandom(4096))
        self.assertFalse(is_lfs_pointer(self.paths.model_file))
        self.assertFalse(is_lfs_pointer(self.root / "missing.keras"))

    def test_loading_a_placeholder_explains_what_to_do(self):
        self.paths.model_file.write_bytes(POINTER)
        with self.assertRaises(ModelLoadError) as raised:
            V7Engine(self.paths).load()
        self.assertIn("git lfs pull", str(raised.exception))

    def test_missing_model_is_reported(self):
        with self.assertRaises(ModelLoadError):
            V7Engine(self.paths).load()


if __name__ == "__main__":
    unittest.main()
