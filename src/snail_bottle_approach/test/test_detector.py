"""Check decoding independently of ROS, network access, and model downloads."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from snail_bottle_approach.detector import BottleDetector, Detection


class FakeSession:
    def __init__(self):
        self.input = SimpleNamespace(name='images', shape=[1, 3, 640, 640],
                                     type='tensor(float)')
        self.output = SimpleNamespace(name='output0', shape=[1, 84, 8400],
                                      type='tensor(float)')
        self.metadata = {'task': 'detect',
                         'names': repr({index: 'bottle' if index == 39 else
                                        'class {}'.format(index)
                                        for index in range(80)})}
        self.result = np.zeros((1, 84, 8400), dtype=np.float32)
        self.last_feed = None

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.output]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.metadata)

    def run(self, names, feed):
        self.last_feed = feed
        return [self.result]

    def predict(self, index, box, score=0.9, class_id=39):
        self.result[0, :4, index] = box
        self.result[0, 4 + class_id, index] = score


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'test-only-model')
    session = FakeSession()
    call = {}

    def create_session(path, sess_options, providers):
        call.update(path=path, options=sess_options, providers=providers)
        return session

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=create_session))
    return model, session, call


def test_cpu_provider_and_limited_threads(fake_runtime):
    model, _, call = fake_runtime
    BottleDetector(model)
    assert call['providers'] == ['CPUExecutionProvider']
    assert call['options'].intra_op_num_threads == 2
    assert call['options'].inter_op_num_threads == 1


def test_missing_model_never_loads_runtime(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    with pytest.raises(FileNotFoundError, match='prepare_bottle_model'):
        BottleDetector(tmp_path / 'not-here.onnx')


def test_missing_runtime_reports_installation(fake_runtime, monkeypatch):
    model, _, _ = fake_runtime
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    with pytest.raises(RuntimeError, match='onnxruntime is missing'):
        BottleDetector(model)


@pytest.mark.parametrize('options', [
    {'confidence': 0.0}, {'confidence': 1.1}, {'confidence': float('nan')},
    {'iou_threshold': 0.0}, {'iou_threshold': float('inf')},
    {'num_threads': 0}, {'num_threads': 1.5}, {'num_threads': True},
])
def test_invalid_options(fake_runtime, options):
    with pytest.raises(ValueError):
        BottleDetector(fake_runtime[0], **options)


@pytest.mark.parametrize('attribute,value', [
    ('input', [1, 3, 'height', 'width']),
    ('input', [1, 640, 640, 3]),
    ('output', [1, 8400, 84]),
    ('output', [1, 85, 8400]),
    ('output', [1, 300, 6]),
])
def test_rejects_incompatible_graph_shapes(fake_runtime, attribute, value):
    model, session, _ = fake_runtime
    getattr(session, attribute).shape = value
    with pytest.raises(ValueError, match='Unsupported ONNX'):
        BottleDetector(model)


@pytest.mark.parametrize('attribute', ['input', 'output'])
def test_rejects_non_float_graph(fake_runtime, attribute):
    model, session, _ = fake_runtime
    getattr(session, attribute).type = 'tensor(float16)'
    with pytest.raises(ValueError, match='Unsupported ONNX'):
        BottleDetector(model)


@pytest.mark.parametrize('metadata', [
    {'task': 'segment'}, {'names': 'not a literal'}, {'names': '{}'},
    {'names': repr({i: 'car' for i in range(80)})},
])
def test_rejects_wrong_class_metadata(fake_runtime, metadata):
    model, session, _ = fake_runtime
    session.metadata = metadata
    with pytest.raises(ValueError):
        BottleDetector(model)


def test_missing_metadata_accepted_for_known_static_shape(fake_runtime):
    model, session, _ = fake_runtime
    session.metadata = {}
    BottleDetector(model)


def test_preprocess_bgr_rgb_normalization_and_padding():
    image = np.full((480, 640, 3), (10, 20, 250), dtype=np.uint8)
    tensor, gain, left, top = BottleDetector._preprocess(image)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert (gain, left, top) == (1.0, 0, 80)
    np.testing.assert_allclose(tensor[0, :, 100, 100], [250 / 255, 20 / 255, 10 / 255])
    np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255)


@pytest.mark.parametrize('image', [
    np.zeros((10, 10), dtype=np.uint8),
    np.zeros((10, 10, 4), dtype=np.uint8),
    np.zeros((10, 10, 3), dtype=np.float32),
    np.zeros((0, 10, 3), dtype=np.uint8), None,
])
def test_invalid_image_rejected(image):
    with pytest.raises(ValueError, match='BGR image'):
        BottleDetector._preprocess(image)


def test_letterbox_inverse_with_resize(fake_runtime):
    model, session, _ = fake_runtime
    detector = BottleDetector(model)
    # 1280x720 -> 640x360, with 140px vertical padding.
    # Original box [400, 100, 600, 500] -> [200, 190, 300, 390].
    session.predict(0, (250, 290, 100, 200))
    detected = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(detected) == 1
    assert detected[0].bbox == pytest.approx((400, 100, 600, 500))
    assert detected[0].confidence == pytest.approx(0.9)


def test_letterbox_inverse_portrait(fake_runtime):
    model, session, _ = fake_runtime
    detector = BottleDetector(model)
    # 480x640 -> width 480 with 80px padding at left.
    session.predict(0, (280, 250, 100, 200))
    detected = detector.detect(np.zeros((640, 480, 3), dtype=np.uint8))
    assert detected[0].bbox == pytest.approx((150, 150, 250, 350))


def test_argmax_class_filter_and_confidence(fake_runtime):
    model, session, _ = fake_runtime
    detector = BottleDetector(model)
    session.predict(0, (100, 100, 20, 40), score=0.7)
    session.predict(0, (100, 100, 20, 40), score=0.95, class_id=0)
    session.predict(1, (200, 100, 20, 40), score=0.54)
    session.predict(2, (300, 100, 20, 40), score=0.8)
    detected = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert len(detected) == 1
    assert detected[0].bbox == pytest.approx((290, 80, 310, 120))


def test_nms_keeps_best_overlap_and_separate_bottle(fake_runtime):
    model, session, _ = fake_runtime
    detector = BottleDetector(model)
    session.predict(0, (100, 100, 60, 100), score=0.7)
    session.predict(1, (101, 101, 60, 100), score=0.9)
    session.predict(2, (400, 100, 60, 100), score=0.8)
    detected = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert len(detected) == 2
    assert detected[0].confidence == pytest.approx(0.9)
    assert detected[1].confidence == pytest.approx(0.8)


def test_clips_boxes_and_discards_outside_frame(fake_runtime):
    model, session, _ = fake_runtime
    detector = BottleDetector(model)
    session.predict(0, (630, 100, 60, 100))
    session.predict(1, (700, 100, 20, 100))
    detected = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert len(detected) == 1
    assert detected[0].bbox == pytest.approx((600, 50, 640, 150))


def test_invalid_predictions_never_become_targets(fake_runtime):
    model, session, _ = fake_runtime
    detector = BottleDetector(model)
    session.predict(0, (100, 100, 60, 100), score=float('nan'))
    session.predict(1, (200, float('inf'), 60, 100))
    session.predict(2, (300, 100, -60, 100))
    session.predict(3, (400, 100, 60, 100), score=1.1)
    assert detector.detect(np.zeros((640, 640, 3), dtype=np.uint8)) == []


def test_unexpected_runtime_output_rejected(fake_runtime):
    model, session, _ = fake_runtime
    detector = BottleDetector(model)
    session.result = np.zeros((1, 84, 1), dtype=np.float32)
    with pytest.raises(RuntimeError, match='unexpected output shape'):
        detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))


def test_detection_is_immutable():
    detection = Detection((1.0, 2.0, 3.0, 4.0), 0.9)
    with pytest.raises(FrozenInstanceError):
        detection.confidence = 0.5
