from array import array
from types import SimpleNamespace
import unittest

import numpy as np

from snail_bottle_approach.image_utils import bgr_image, depth_metres


def image(encoding, width, height, step, data, bigendian=False):
    return SimpleNamespace(encoding=encoding, width=width, height=height,
                           step=step, data=array('B', data), is_bigendian=bigendian)


class ImageTests(unittest.TestCase):
    def test_rgb_padding_and_channel_order(self):
        message = image('rgb8', 1, 2, 4, bytes([10, 20, 30, 99, 40, 50, 60, 99]))
        np.testing.assert_array_equal(bgr_image(message), [[[30, 20, 10]], [[60, 50, 40]]])

    def test_depth_byte_order_and_padding(self):
        for endian, big in (('<', False), ('>', True)):
            raw = np.array([[1000, 2000, 65535]], dtype=endian + 'u2').tobytes()
            message = image('16UC1', 2, 1, 6, raw, big)
            np.testing.assert_allclose(depth_metres(message), [[1.0, 2.0]])

    def test_float_metres(self):
        raw = np.array([[1.25, np.nan]], dtype='>f4').tobytes()
        result = depth_metres(image('32FC1', 2, 1, 8, raw, True))
        self.assertAlmostEqual(result[0, 0], 1.25)
        self.assertTrue(np.isnan(result[0, 1]))

    def test_malformed_image_rejected(self):
        for message in (image('rgb8', 2, 1, 3, b'123'),
                        image('rgb8', 1, 1, 3, b'12'),
                        image('8UC1', 1, 1, 1, b'1')):
            with self.assertRaises(ValueError):
                bgr_image(message)
