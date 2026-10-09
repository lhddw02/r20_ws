"""Decode ROS images without cv_bridge, preserving row padding and byte order."""

import numpy as np


def _view(message, dtype, channels=1):
    height, width, step = message.height, message.width, message.step
    dtype = np.dtype(dtype)
    row_bytes = width * channels * dtype.itemsize
    if height <= 0 or width <= 0 or step < row_bytes:
        raise ValueError('Invalid image dimensions or row stride')
    if len(message.data) != height * step:
        raise ValueError('Image byte count does not match height and row stride')
    shape = (height, width, channels) if channels > 1 else (height, width)
    strides = ((step, channels * dtype.itemsize, dtype.itemsize) if channels > 1
               else (step, dtype.itemsize))
    return np.ndarray(shape, dtype=dtype, buffer=message.data, strides=strides)


def bgr_image(message):
    if message.encoding not in ('bgr8', 'rgb8'):
        raise ValueError(f'Unsupported color encoding {message.encoding!r}')
    pixels = _view(message, 'u1', 3)
    if message.encoding == 'rgb8':
        pixels = pixels[:, :, ::-1]
    return np.ascontiguousarray(pixels)


def depth_metres(message):
    endian = '>' if message.is_bigendian else '<'
    if message.encoding == '16UC1':
        # RealSense ROS depth images use millimetres for 16UC1.
        return _view(message, endian + 'u2').astype(np.float32) * 0.001
    if message.encoding == '32FC1':
        return _view(message, endian + 'f4').astype(np.float32)
    raise ValueError(f'Unsupported depth encoding {message.encoding!r}')


def stamp_seconds(message):
    stamp = message.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9
