"""Exercise real ROS image/topic/service boundaries without a camera or model.

The detector is the only substitute. ROS messages, subscriptions, command
timers, image conversion, depth estimation, and the inference thread are real.
The tests require a sourced ROS 2 Humble environment.
"""

import threading
import time
import uuid

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy', reason='ROS 2 is not installed')
pytest.importorskip('cv2', reason='OpenCV is not installed')

from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from snail_bottle_approach.detector import Detection
from snail_bottle_approach.node import BottleApproach


class FakeDetector:
    """Capture a detection snapshot before an optional blocked inference."""

    def __init__(self, delay=0.0):
        self.detections = [Detection((60.0, 30.0, 100.0, 90.0), 0.95)]
        self.delay = delay
        self.calls = 0
        self.block_next = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def detect(self, bgr):
        assert bgr.shape == (120, 160, 3)
        assert bgr.dtype == np.uint8
        detections = list(self.detections)
        self.calls += 1
        if self.block_next:
            self.block_next = False
            self.entered.set()
            # A bounded wait also ensures an assertion failure cannot leak a
            # permanently blocked thread into the next test.
            assert self.release.wait(timeout=4.0)
        if self.delay:
            time.sleep(self.delay)
        return detections

    def block(self):
        self.entered.clear()
        self.release.clear()
        self.block_next = True


class CameraHarness:
    """Publish synthetic RGB-D frames and observe actual velocity messages."""

    def __init__(self, enable_motion=True, initial_connected=True,
                 detector_delay=0.0, **overrides):
        self.detector = FakeDetector(delay=detector_delay)
        self.node = self.peer = self.executor = None
        self.initialized = self.closed = False
        self.commands = []
        self.command_audit = []
        self.last_publish = 0.0
        self.frame_count = 0
        self.initial_connected = initial_connected
        self.namespace = '/bottle_test_' + uuid.uuid4().hex[:10]
        self.topics = {key: self.namespace + '/' + key for key in (
            'color', 'depth', 'info', 'cmd', 'connected')}
        parameters = {
            'enable_motion': 'true' if enable_motion else 'false',
            'color_topic': self.topics['color'],
            'depth_topic': self.topics['depth'],
            'camera_info_topic': self.topics['info'],
            'cmd_topic': self.topics['cmd'],
            'connected_topic': self.topics['connected'],
            'frame_timeout': 0.25,
            'connected_timeout': 0.50,
            'stop_distance': 0.60,
            'confirmations': 3,
        }
        parameters.update(overrides)
        args = ['--ros-args', '-r', '__ns:=' + self.namespace]
        for name, value in parameters.items():
            args.extend(['-p', f'{name}:={value}'])
        try:
            rclpy.init(args=args)
            self.initialized = True
            self.node = BottleApproach(detector=self.detector)
            publish_command = self.node.publish_command

            def audited_command(linear, angular):
                # Record source freshness when the real control timer publishes,
                # before DDS delivery can add observation latency.
                if linear != 0.0 or angular != 0.0:
                    stamp = self.node.last_result_stamp
                    self.command_audit.append({
                        'frames_processed': self.node.frames_processed,
                        'capture_fresh': self.node.capture_fresh(stamp),
                        'capture_age': self.node.get_clock().now().nanoseconds * 1e-9 - stamp,
                        'receive_age': time.monotonic() - self.node.last_result_received,
                    })
                publish_command(linear, angular)

            self.node.publish_command = audited_command
            self.peer = rclpy.create_node('synthetic_camera')
            self.publishers = {}
            for key, message_type in (
                    ('color', Image), ('depth', Image), ('info', CameraInfo),
                    ('connected', Bool)):
                self.publishers[key] = self.peer.create_publisher(
                    message_type, self.topics[key], 1)
            self.command_sub = self.peer.create_subscription(
                Twist, self.topics['cmd'], self.commands.append, 50)
            private = self.node.get_fully_qualified_name()
            self.clients = {action: self.peer.create_client(Trigger, private + '/' + action)
                            for action in ('start', 'stop')}
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.executor.add_node(self.peer)
        except Exception:
            self.close()
            raise

    def frame(self, distance=1.20, *, age=0.0, stamp=None,
              depth_frame='camera_color_optical_frame', invalid_info=False,
              depth_delta=0.0, streams=('info', 'color', 'depth')):
        now = self.peer.get_clock().now().nanoseconds
        nanoseconds = int(now - age * 1e9) if stamp is None else stamp

        def assign_stamp(header, value):
            header.stamp.sec, header.stamp.nanosec = divmod(int(value), 1_000_000_000)

        info = CameraInfo()
        assign_stamp(info.header, nanoseconds)
        info.header.frame_id = 'camera_color_optical_frame'
        info.width, info.height = 160, 120
        info.distortion_model = 'plumb_bob'
        info.k = [0.0 if invalid_info else 100.0, 0.0, 80.0,
                  0.0, 100.0, 60.0, 0.0, 0.0, 1.0]
        info.p = [100.0, 0.0, 80.0, 0.0,
                  0.0, 100.0, 60.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        color = Image()
        assign_stamp(color.header, nanoseconds)
        color.header.frame_id = 'camera_color_optical_frame'
        color.width, color.height = 160, 120
        color.encoding, color.step = 'bgr8', 160 * 3
        color.data = np.zeros((120, 160, 3), dtype=np.uint8).tobytes()
        depth = Image()
        assign_stamp(depth.header, nanoseconds + depth_delta * 1e9)
        depth.header.frame_id = depth_frame
        depth.width, depth.height = 160, 120
        depth.encoding, depth.step = '16UC1', 160 * 2
        depth.data = np.full((120, 160), int(distance * 1000), dtype='<u2').tobytes()
        messages = {'info': info, 'color': color, 'depth': depth}
        for stream in streams:
            self.publishers[stream].publish(messages[stream])
        self.frame_count += 1
        return nanoseconds

    def step(self, *, frames=False, connected=True, distance=1.20, **frame_options):
        now = time.monotonic()
        if now - self.last_publish >= 0.04:
            self.last_publish = now
            if connected is not None:
                message = Bool()
                message.data = connected
                self.publishers['connected'].publish(message)
            if frames:
                self.frame(distance, **frame_options)
        self.executor.spin_once(timeout_sec=0.005)

    def run(self, duration, **options):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.step(**options)

    def until(self, predicate, timeout=2.0, **options):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.step(**options)
            if predicate():
                return
        assert predicate(), 'Timed out waiting for ROS bottle-approach condition'

    def discover(self):
        self.until(lambda: all(pub.get_subscription_count() >= 1
                               for pub in self.publishers.values())
                   and self.peer.count_publishers(self.topics['cmd']) >= 1
                   and all(client.service_is_ready() for client in self.clients.values()),
                   connected=self.initial_connected)
        self.until(lambda: self.node.base_received is not None,
                   connected=self.initial_connected)

    def service(self, action, **pump_options):
        future = self.clients[action].call_async(Trigger.Request())
        self.until(future.done, **pump_options)
        return future.result()

    def moving(self):
        return bool(self.commands) and (self.commands[-1].linear.x > 0.0
                                       or self.commands[-1].angular.z != 0.0)

    def stopped(self):
        return bool(self.commands) and self.commands[-1].linear.x == 0.0 \
            and self.commands[-1].angular.z == 0.0

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.detector.release.set()
        if self.node is not None:
            self.node.stop()
        if self.executor is not None:
            self.executor.shutdown()
        if self.peer is not None:
            self.peer.destroy_node()
        if self.node is not None:
            self.node.destroy_node()
        if self.initialized and rclpy.ok():
            rclpy.shutdown()


@pytest.fixture
def camera_factory():
    harnesses = []

    def create(**options):
        harness = CameraHarness(**options)
        harnesses.append(harness)
        harness.discover()
        return harness

    yield create
    for harness in reversed(harnesses):
        harness.close()


def test_detection_only_streams_images_but_never_commands_motion(camera_factory):
    harness = camera_factory(enable_motion=False)
    assert not harness.service('start').success
    harness.until(lambda: harness.node.frames_processed >= 3, frames=True)
    harness.run(0.15, frames=True)
    assert harness.commands
    assert all(message.linear.x == message.angular.z == 0.0
               for message in harness.commands)
    assert not harness.node.armed


def test_camera_right_turn_forward_approach_and_latched_stop(camera_factory):
    harness = camera_factory()
    harness.detector.detections = [Detection((110.0, 30.0, 150.0, 90.0), 0.95)]
    assert harness.service('start').success
    harness.until(lambda: harness.moving(), frames=True)
    assert harness.commands[-1].linear.x == 0.0
    assert harness.commands[-1].angular.z < 0.0

    # Moving to the image center requires new target confirmation and then
    # moves forward. The turn is not inferred from a hand-built command.
    harness.detector.detections = [Detection((60.0, 30.0, 100.0, 90.0), 0.95)]
    harness.until(lambda: bool(harness.commands)
                  and harness.commands[-1].linear.x > 0.0, frames=True)
    assert harness.commands[-1].angular.z == 0.0
    assert harness.commands[-1].linear.x <= 0.08
    harness.until(lambda: harness.node.policy.state == 'REACHED',
                  frames=True, distance=0.45)
    harness.until(harness.stopped, frames=True, distance=0.45)
    mark = len(harness.commands)
    harness.run(0.30, frames=True, distance=1.50)
    assert harness.node.policy.state == 'REACHED'
    assert all(message.linear.x == message.angular.z == 0.0
               for message in harness.commands[mark:])


@pytest.mark.parametrize('bad_frame', [
    {'age': 1.0},
    {'depth_frame': 'camera_depth_optical_frame'},
    {'invalid_info': True},
    {'depth_delta': -0.15},
    {'distance': 0.0},
])
def test_invalid_sensor_inputs_stop_existing_motion(camera_factory, bad_frame):
    harness = camera_factory()
    assert harness.service('start').success
    harness.until(harness.moving, frames=True)
    harness.run(0.45, frames=True, **bad_frame)
    assert harness.stopped()
    mark = len(harness.commands)
    harness.run(0.15, frames=True, **bad_frame)
    assert all(message.linear.x == message.angular.z == 0.0
               for message in harness.commands[mark:])


def test_camera_loss_and_replayed_timestamps_cannot_keep_motion_alive(camera_factory):
    harness = camera_factory()
    assert harness.service('start').success
    harness.until(harness.moving, frames=True)
    harness.run(0.45)
    assert harness.stopped()
    stamp = harness.frame()
    harness.run(0.40, frames=True, stamp=stamp)
    assert harness.stopped()
    mark = len(harness.commands)
    harness.run(0.15, frames=True, stamp=stamp)
    assert all(message.linear.x == message.angular.z == 0.0
               for message in harness.commands[mark:])


def test_base_loss_disarms_and_recovery_requires_explicit_start(camera_factory):
    harness = camera_factory()
    assert harness.service('start').success
    harness.until(harness.moving, frames=True)
    harness.until(lambda: not harness.node.armed and harness.stopped(),
                  frames=True, connected=False)
    mark = len(harness.commands)
    harness.run(0.35, frames=True, connected=True)
    assert not harness.node.armed
    assert all(message.linear.x == message.angular.z == 0.0
               for message in harness.commands[mark:])
    assert harness.service('start').success
    harness.until(harness.moving, frames=True)


def test_initial_unready_base_waits_and_first_connection_permits_motion(camera_factory):
    harness = camera_factory(initial_connected=False)
    assert not harness.service('start', connected=False).success
    harness.run(0.15, frames=True, connected=False)
    assert harness.node.armed
    assert harness.stopped()
    # Startup telemetry may initially be false while the serial bridge opens.
    # This is an initial readiness wait, rather than a lost established link.
    harness.until(harness.moving, frames=True, connected=True)


def test_blocked_inference_stops_and_discards_pending_frame_backlog(camera_factory):
    harness = camera_factory()
    assert harness.service('start').success
    harness.until(harness.moving, frames=True)
    harness.detector.block()
    harness.until(harness.detector.entered.is_set, frames=True)
    baseline = harness.detector.calls
    start_frame_count = harness.frame_count
    harness.run(0.50, frames=True)
    assert harness.stopped()
    assert harness.frame_count - start_frame_count >= 8
    assert harness.detector.calls == baseline
    # There is one replaceable pending frame, rather than a FIFO of all frames
    # published while the detector was busy. No fresh input follows release.
    harness.detector.release.set()
    harness.run(0.35)
    assert harness.detector.calls <= baseline + 1
    assert harness.stopped()


@pytest.mark.parametrize('inference_delay', [0.28, 0.35])
def test_slow_fresh_inference_accumulates_confirmations_without_extending_timeout(
        camera_factory, inference_delay):
    harness = camera_factory(frame_timeout=0.50, detector_delay=inference_delay)
    assert harness.service('start').success

    # A 25 Hz camera keeps publishing while inference takes most of the 0.5 s
    # validity window. There are short zero-command gaps between fresh results;
    # those must not repeatedly erase the still-consistent target confirmations.
    harness.until(harness.moving, timeout=4.0, frames=True)
    assert harness.command_audit
    assert harness.command_audit[0]['frames_processed'] >= 3
    assert harness.commands[-1].linear.x > 0.0
    assert harness.frame_count >= 10
    harness.run(0.75, frames=True)

    # An actual source outage must stop. The next observation's original
    # timestamp must then expose the source gap and restart confirmation.
    harness.run(0.85)
    assert harness.stopped()
    assert not harness.node.observation_fresh(time.monotonic())
    baseline = harness.node.frames_processed
    harness.until(lambda: harness.node.frames_processed >= baseline + 1,
                  timeout=2.0, frames=True)
    assert harness.stopped()
    assert harness.node.policy.state == 'CONFIRMING'
    harness.until(lambda: harness.node.frames_processed >= baseline + 2,
                  timeout=2.0, frames=True)
    assert harness.stopped()
    harness.until(harness.moving, timeout=2.0, frames=True)
    assert harness.node.frames_processed >= baseline + 3

    # Confirmation retention never grants motion using an expired source image.
    assert all(entry['capture_fresh'] for entry in harness.command_audit)
    assert all(-0.05 <= entry['capture_age'] <= 0.50
               and 0.0 <= entry['receive_age'] < 0.50
               for entry in harness.command_audit)


def test_inference_slower_than_source_timeout_never_commands_motion(camera_factory):
    harness = camera_factory(frame_timeout=0.50, detector_delay=0.60)
    assert harness.service('start').success
    harness.run(2.25, frames=True)
    assert harness.detector.calls >= 3
    assert harness.commands
    assert harness.node.frames_processed == 0
    assert not harness.command_audit
    assert all(message.linear.x == message.angular.z == 0.0
               for message in harness.commands)


def test_stop_start_cancels_a_result_from_the_previous_approach(camera_factory):
    harness = camera_factory(frame_timeout=0.80, confirmations=1)
    assert harness.service('start').success
    harness.detector.block()
    harness.until(harness.detector.entered.is_set, frames=True, distance=0.45)
    assert harness.service('stop').success
    assert not harness.node.armed
    assert harness.service('start').success
    harness.detector.release.set()
    harness.run(0.20)
    assert harness.node.policy.state == 'WAITING'
    assert harness.stopped()
    harness.until(harness.moving, frames=True, distance=1.20)


def test_confirmations_require_three_distinct_complete_rgbd_observations(camera_factory):
    harness = camera_factory(frame_timeout=0.80)
    assert harness.service('start').success
    harness.frame()
    harness.until(lambda: harness.node.frames_processed == 1)
    # A new RGB image must wait for a new depth image. Reusing the previous
    # depth would manufacture an extra confirmation from two observations.
    color_stamp = harness.frame(streams=('color',))
    harness.run(0.10)
    assert harness.node.frames_processed == 1
    harness.frame(stamp=color_stamp, streams=('depth',))
    harness.until(lambda: harness.node.frames_processed == 2)
    assert harness.node.policy.state == 'CONFIRMING'
    assert harness.stopped()
    harness.frame()
    harness.until(harness.moving)


def test_operator_stop_remains_visible_while_perception_continues(camera_factory):
    harness = camera_factory()
    assert harness.service('start').success
    harness.until(harness.moving, frames=True)
    assert harness.service('stop').success
    harness.run(0.25, frames=True)
    assert not harness.node.armed
    assert harness.stopped()
    assert harness.node.state_reason == 'STOPPED'
    harness.run(0.35)
    assert harness.node.state_reason == 'STOPPED'
    assert harness.service('start').success
    harness.until(harness.moving, frames=True)


def test_bottle_inside_camera_to_front_offset_stops_and_latches(camera_factory):
    harness = camera_factory(camera_to_front=0.30, stop_distance=0.40)
    assert harness.service('start').success
    harness.until(lambda: harness.node.policy.state == 'REACHED',
                  frames=True, distance=0.25)
    assert harness.stopped()
    harness.run(0.20, frames=True, distance=1.20)
    assert harness.node.policy.state == 'REACHED'
    assert harness.stopped()


def test_other_command_publisher_inhibits_start_and_disarms_motion(camera_factory):
    harness = camera_factory()
    assert harness.service('start').success
    harness.until(harness.moving, frames=True)
    foreign = harness.peer.create_publisher(Twist, harness.topics['cmd'], 1)
    harness.until(lambda: harness.node.count_publishers(harness.topics['cmd']) > 1)
    assert not harness.service('start').success
    harness.until(lambda: not harness.node.armed and harness.stopped(), frames=True)
    harness.peer.destroy_publisher(foreign)
    harness.until(lambda: harness.node.count_publishers(harness.topics['cmd']) == 1)
    harness.run(0.15, frames=True)
    assert not harness.node.armed
    assert harness.service('start').success
    harness.until(harness.moving, frames=True)
