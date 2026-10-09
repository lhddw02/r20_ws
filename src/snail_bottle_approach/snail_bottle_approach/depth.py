"""Extract a conservative target estimate from aligned depth in metres.

The input must already be registered to the color image. A robust median of
the central half of the box avoids box-edge background. A broad or bimodal
central depth distribution is rejected, rather than picking a convenient near
cluster. Transparent bottles can still return uniformly plausible background
depth; this estimator does not solve that sensor limitation or identify water.
"""

import math
from typing import Optional, Sequence

import numpy as np

from .approach_policy import Target


def estimate_target(depth: np.ndarray, bbox: Sequence[float], fx: float,
                    cx: float, confidence: float, min_depth: float = 0.15,
                    max_depth: float = 5.0, *, min_valid_fraction: float = 0.25,
                    min_valid_pixels: int = 20, center_fraction: float = 0.50,
                    max_relative_spread: float = 0.25,
                    max_absolute_spread: float = 0.20) -> Optional[Target]:
    """Return a target, or None if depth/geometry are insufficient.

    Spread means the 90th minus 10th percentile, in metres, and must pass both
    absolute and relative limits. At least 20 valid pixels and 25% coverage of
    the central window are required by default. Zeros, NaNs, infinities and
    out-of-range pixels are invalid. Bounding boxes must lie inside the image.
    """
    try:
        if (not isinstance(depth, np.ndarray) or depth.ndim != 2
                or not np.issubdtype(depth.dtype, np.number)
                or np.issubdtype(depth.dtype, np.complexfloating)):
            return None
        height, width = depth.shape
        x1, y1, x2, y2 = (float(value) for value in bbox)
        scalars = (x1, y1, x2, y2, fx, cx, confidence, min_depth, max_depth,
                   min_valid_fraction, center_fraction, max_relative_spread,
                   max_absolute_spread)
        if not all(math.isfinite(value) for value in scalars):
            return None
        if (height == 0 or width == 0 or fx <= 0 or not 0 <= cx < width
                or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height
                or not 0 < confidence <= 1 or not 0 < min_depth < max_depth
                or not 0 < min_valid_fraction <= 1
                or not 0 < center_fraction <= 1
                or not isinstance(min_valid_pixels, int)
                or isinstance(min_valid_pixels, bool) or min_valid_pixels < 1
                or max_relative_spread <= 0 or max_absolute_spread <= 0):
            return None
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        half_width = (x2 - x1) * center_fraction / 2
        half_height = (y2 - y1) * center_fraction / 2
        left = max(0, int(math.ceil(center_x - half_width)))
        right = min(width, int(math.floor(center_x + half_width)))
        top = max(0, int(math.ceil(center_y - half_height)))
        bottom = min(height, int(math.floor(center_y + half_height)))
        roi = depth[top:bottom, left:right]
        if roi.size == 0:
            return None
        valid = roi[np.isfinite(roi) & (roi >= min_depth) & (roi <= max_depth)]
        if valid.size < min_valid_pixels or valid.size / roi.size < min_valid_fraction:
            return None
        distance = float(np.median(valid))
        low, high = np.percentile(valid, [10, 90])
        spread = float(high - low)
        if (spread > max_absolute_spread
                or spread > distance * max_relative_spread):
            return None
        bearing = math.atan2(center_x - cx, fx)
        return Target((x1, y1, x2, y2), distance, bearing, float(confidence))
    except (TypeError, ValueError, OverflowError):
        return None
