import unittest

from v7_segmenter.imaging.viewport import Viewport


def make(image=(1000, 500), view=(500, 500)) -> Viewport:
    viewport = Viewport()
    viewport.set_view(*view)
    viewport.set_image(*image)
    return viewport


class ViewportTest(unittest.TestCase):
    def test_fit_shows_the_whole_image(self):
        region = make().visible_region()
        self.assertEqual((region.x0, region.y0, region.x1, region.y1), (0, 0, 1000, 500))
        self.assertLessEqual(region.screen_w, 500)
        self.assertLessEqual(region.screen_h, 500)

    def test_zoom_keeps_the_point_under_the_cursor(self):
        viewport = make()
        before = viewport.screen_to_image(100, 200)
        viewport.zoom_by(2.0, (100, 200))
        after = viewport.screen_to_image(100, 200)
        self.assertAlmostEqual(before[0], after[0], places=6)
        self.assertAlmostEqual(before[1], after[1], places=6)

    def test_screen_image_round_trip(self):
        viewport = make()
        viewport.zoom_by(3.0, (250, 250))
        x, y = viewport.screen_to_image(123, 456)
        sx, sy = viewport.image_to_screen(x, y)
        self.assertAlmostEqual(sx, 123, places=6)
        self.assertAlmostEqual(sy, 456, places=6)

    def test_pan_stops_at_the_image_edge(self):
        viewport = make()
        viewport.set_zoom(4.0)
        viewport.pan(10_000, 10_000)
        self.assertEqual((viewport.center_x, viewport.center_y), (0.0, 0.0))

    def test_zoomed_region_stays_inside_the_image(self):
        viewport = make()
        viewport.set_zoom(5.0, (250, 250))
        region = viewport.visible_region()
        self.assertTrue(0 <= region.x0 < region.x1 <= 1000)
        self.assertTrue(0 <= region.y0 < region.y1 <= 500)

    def test_resizing_the_window_refits_until_the_user_zooms(self):
        viewport = make()
        viewport.set_view(1000, 1000)
        self.assertAlmostEqual(viewport.zoom, viewport.fit_zoom())
        viewport.zoom_by(2.0)
        zoom = viewport.zoom
        viewport.set_view(800, 800)
        self.assertEqual(viewport.zoom, zoom)

    def test_no_image_means_nothing_visible(self):
        self.assertIsNone(Viewport().visible_region())


if __name__ == "__main__":
    unittest.main()
