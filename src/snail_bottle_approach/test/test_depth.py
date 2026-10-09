"""Verify depth rejection and optical-coordinate ranging on synthetic frames."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snail_bottle_approach.depth import estimate_target


class DepthTests(unittest.TestCase):
    bbox = (10, 10, 50, 50)

    def estimate(self, depth=None, bbox=None, **kwargs):
        if depth is None:
            depth = np.full((80, 100), 1.2, dtype=np.float32)
        if bbox is None:
            bbox = self.bbox
        return estimate_target(depth, bbox, kwargs.pop('fx', 100.0),
                               kwargs.pop('cx', 50.0),
                               kwargs.pop('confidence', 0.9), **kwargs)

    def test_median_and_camera_bearing(self):
        item = self.estimate()
        self.assertIsNotNone(item)
        self.assertAlmostEqual(item.distance, 1.2, places=6)
        self.assertAlmostEqual(item.bearing, math.atan2(-20, 100))
        self.assertEqual(item.bbox, (10.0, 10.0, 50.0, 50.0))
        self.assertEqual(item.confidence, 0.9)
        right = self.estimate(bbox=(60, 10, 100, 50))
        self.assertGreater(right.bearing, 0)

    def test_box_edge_background_is_excluded(self):
        depth = np.full((80, 100), 3.0)
        depth[20:40, 20:40] = 0.9
        self.assertAlmostEqual(self.estimate(depth).distance, 0.9)

    def test_small_noise_and_sparse_outliers_are_robust(self):
        depth = np.full((80, 100), 1.0)
        depth[20:40, 20:40] = np.linspace(0.99, 1.01, 400).reshape(20, 20)
        depth[20, 20] = 4.0
        self.assertAlmostEqual(self.estimate(depth).distance, 1.0, places=3)

    def test_bimodal_background_depth_is_rejected(self):
        depth = np.full((80, 100), 2.0)
        depth[20:26, 20:40] = 0.8
        self.assertIsNone(self.estimate(depth))

    def test_absolute_spread_limit_rejects_far_ambiguous_surface(self):
        depth = np.full((80, 100), 3.0)
        depth[20:40, 20:40] = np.linspace(2.8, 3.2, 400).reshape(20, 20)
        self.assertIsNone(self.estimate(depth))

    def test_relative_spread_limit_rejects_near_ambiguous_surface(self):
        depth = np.full((80, 100), 0.3)
        depth[20:40, 20:40] = np.linspace(0.2, 0.4, 400).reshape(20, 20)
        self.assertIsNone(self.estimate(depth))

    def test_insufficient_valid_fraction_is_rejected(self):
        depth = np.zeros((80, 100))
        depth[20:24, 20:40] = 1.0
        self.assertIsNone(self.estimate(depth))
        depth[24, 20:40] = 1.0
        self.assertAlmostEqual(self.estimate(depth).distance, 1.0)

    def test_minimum_valid_pixel_count_is_required(self):
        depth = np.ones((80, 100))
        self.assertIsNone(self.estimate(depth, bbox=(0, 0, 8, 8)))
        self.assertIsNotNone(self.estimate(depth, bbox=(0, 0, 12, 12)))
        self.assertIsNotNone(self.estimate(depth, min_valid_pixels=400))
        self.assertIsNone(self.estimate(depth, min_valid_pixels=401))

    def test_nan_inf_zero_and_out_of_range_are_invalid(self):
        for value in (math.nan, math.inf, 0.0, -1.0, 0.149, 5.01):
            with self.subTest(value=value):
                self.assertIsNone(self.estimate(np.full((80, 100), value)))

    def test_depth_range_endpoints_are_accepted(self):
        for value in (0.15, 5.0):
            with self.subTest(value=value):
                self.assertAlmostEqual(self.estimate(np.full((80, 100), value)).distance,
                                       value)

    def test_image_shapes_and_dtypes_are_validated(self):
        for depth in (np.ones((80, 100, 3)), np.ones((0, 0)),
                      np.ones((80, 100), dtype=np.complex128),
                      np.full((80, 100), '1.2'), [[1.2]]):
            with self.subTest(shape=getattr(depth, 'shape', None)):
                self.assertIsNone(self.estimate(depth))

    def test_outside_and_invalid_boxes_are_rejected(self):
        for box in ((-1, 0, 20, 20), (0, 0, 101, 20), (0, 0, 20, 81),
                    (10, 10, 10, 50), (10, 10, 5, 50),
                    (0, 0, math.nan, 50), (0, 0, 20)):
            with self.subTest(box=box):
                self.assertIsNone(self.estimate(bbox=box))

    def test_invalid_intrinsics_confidence_and_configuration_are_rejected(self):
        for kwargs in ({'fx': 0}, {'fx': math.nan}, {'cx': math.inf},
                       {'cx': -1}, {'cx': 100}, {'confidence': 0},
                       {'confidence': 1.1}, {'min_valid_fraction': 0},
                       {'min_valid_pixels': 0}, {'center_fraction': 1.1},
                       {'max_relative_spread': math.inf},
                       {'max_depth': 0.1}):
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(self.estimate(**kwargs))


if __name__ == '__main__':
    unittest.main()
