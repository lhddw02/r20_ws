"""Small COCO bottle detector without a PyTorch dependency at runtime.

The supported model is YOLOv8n exported as a static 640px ONNX detection
graph, without embedded NMS. COCO's bottle category includes bottles other
than water bottles; these weights cannot identify a bottle's contents.
"""

import ast
from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """A bottle box in the original image's pixel coordinates."""

    bbox: Tuple[float, float, float, float]
    confidence: float


class BottleDetector:
    """Run a previously prepared ONNX model on CPU.

    Inference is synchronous. Call ``detect`` from a worker thread rather
    than the ROS command timer so a slow frame cannot delay stop commands.
    The constructor never downloads weights or installs dependencies.
    """

    INPUT_SIZE = 640
    BOTTLE_CLASS = 39
    OUTPUT_SHAPE = (1, 84, 8400)

    def __init__(self, model_path, confidence=0.55, iou_threshold=0.45,
                 num_threads=2):
        if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
            raise ValueError('confidence must be finite and in (0, 1]')
        if not math.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
            raise ValueError('iou_threshold must be finite and in (0, 1]')
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) \
                or num_threads < 1:
            raise ValueError('num_threads must be a positive integer')
        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                'Bottle ONNX model does not exist: {}. Run '
                'scripts/prepare_bottle_model.py on a development computer '
                'and copy the exported .onnx file to the Jetson.'.format(model_path))
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                'onnxruntime is missing. Install the CPU ONNX Runtime package '
                'in the Python environment used by this ROS node.') from error

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options,
            providers=['CPUExecutionProvider'])
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or tuple(inputs[0].shape) != (1, 3, 640, 640) \
                or inputs[0].type != 'tensor(float)':
            raise ValueError('Unsupported ONNX input: expected one float32 '
                             'input shaped [1, 3, 640, 640]')
        if len(outputs) != 1 or tuple(outputs[0].shape) != self.OUTPUT_SHAPE \
                or outputs[0].type != 'tensor(float)':
            raise ValueError('Unsupported ONNX output: expected one float32 '
                             'output shaped [1, 84, 8400] from YOLOv8 COCO '
                             'detection with nms=False')
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self._validate_metadata(self._session.get_modelmeta().custom_metadata_map)

    @staticmethod
    def _validate_metadata(metadata):
        if metadata.get('task', 'detect') != 'detect':
            raise ValueError('The ONNX model must be an object detection model')
        if 'names' not in metadata:
            return
        try:
            names = ast.literal_eval(metadata['names'])
            if isinstance(names, dict):
                names = {int(key): value for key, value in names.items()}
                valid = (set(names) == set(range(80))
                         and names[39] == 'bottle')
            elif isinstance(names, (list, tuple)):
                valid = len(names) == 80 and names[39] == 'bottle'
            else:
                valid = False
        except (ValueError, TypeError, SyntaxError):
            valid = False
        if not valid:
            raise ValueError('Model class metadata must contain 80 COCO classes '
                             'with class 39 named bottle')

    @classmethod
    def _preprocess(cls, bgr):
        if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 \
                or bgr.ndim != 3 or bgr.shape[2] != 3 \
                or bgr.shape[0] == 0 or bgr.shape[1] == 0:
            raise ValueError('Expected a nonempty uint8 HxWx3 BGR image')
        height, width = bgr.shape[:2]
        gain = min(cls.INPUT_SIZE / width, cls.INPUT_SIZE / height)
        resized_width = max(1, round(width * gain))
        resized_height = max(1, round(height * gain))
        left = (cls.INPUT_SIZE - resized_width) // 2
        top = (cls.INPUT_SIZE - resized_height) // 2
        resized = cv2.resize(bgr, (resized_width, resized_height),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.full((cls.INPUT_SIZE, cls.INPUT_SIZE, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = resized
        # RGB, NCHW, float32 in [0, 1]; slicing reverses the BGR channels.
        tensor = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor *= 1.0 / 255.0
        return tensor, gain, left, top

    def detect(self, bgr) -> List[Detection]:
        tensor, gain, left, top = self._preprocess(bgr)
        result = self._session.run(
            [self._output_name], {self._input_name: tensor})
        if len(result) != 1 or not isinstance(result[0], np.ndarray) \
                or result[0].shape != self.OUTPUT_SHAPE:
            raise RuntimeError('ONNX inference returned an unexpected output shape')
        predictions = result[0][0].T
        scores = predictions[:, 4:]
        # Require bottle to be the winning class. A dog with a secondary
        # bottle score must not become a bottle just because other classes
        # have been discarded before computing the winning class.
        finite = np.isfinite(predictions).all(axis=1)
        mask = (finite & (np.argmax(scores, axis=1) == self.BOTTLE_CLASS)
                & (scores[:, self.BOTTLE_CLASS] >= self.confidence)
                & (scores[:, self.BOTTLE_CLASS] <= 1.0)
                & (predictions[:, 2] > 0.0) & (predictions[:, 3] > 0.0))
        candidates = predictions[mask]
        if not len(candidates):
            return []
        boxes = np.empty((len(candidates), 4), dtype=np.float32)
        boxes[:, :2] = candidates[:, :2] - candidates[:, 2:4] / 2.0
        boxes[:, 2:] = candidates[:, :2] + candidates[:, 2:4] / 2.0
        boxes[:, (0, 2)] = (boxes[:, (0, 2)] - left) / gain
        boxes[:, (1, 3)] = (boxes[:, (1, 3)] - top) / gain
        height, width = bgr.shape[:2]
        boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0.0, width)
        boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0.0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        confidences = candidates[valid, 4 + self.BOTTLE_CLASS]
        keep = self._nms(boxes, confidences)
        return [Detection(tuple(float(value) for value in boxes[index]),
                          float(confidences[index])) for index in keep]

    def _nms(self, boxes, scores):
        order = np.argsort(-scores, kind='stable')
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = []
        while len(order):
            index = int(order[0])
            keep.append(index)
            rest = order[1:]
            if not len(rest):
                break
            lower = np.maximum(boxes[index, :2], boxes[rest, :2])
            upper = np.minimum(boxes[index, 2:], boxes[rest, 2:])
            size = np.maximum(0.0, upper - lower)
            intersection = size[:, 0] * size[:, 1]
            iou = intersection / (areas[index] + areas[rest] - intersection)
            order = rest[iou <= self.iou_threshold]
        return keep
