"""RGB-D bottle perception with a separate, monotonic velocity publish loop."""

from array import array
from dataclasses import replace
import json
import math
import signal
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from snail_bottle_approach.approach_policy import ApproachPolicy
from snail_bottle_approach.depth import estimate_target
from snail_bottle_approach.detector import BottleDetector
from snail_bottle_approach.image_utils import bgr_image, depth_metres, stamp_seconds


class BottleApproach(Node):
    def __init__(self, detector=None):
        super().__init__('bottle_approach')
        defaults = {
            'model_path': '', 'enable_motion': False,
            'color_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/aligned_depth_to_color/camera_info',
            'cmd_topic': '/bottle_approach/cmd_vel', 'connected_topic': '/r20/connected',
            'require_base_connected': True, 'connected_timeout': 2.0,
            'frame_timeout': 0.5, 'sync_slop': 0.08, 'confidence': 0.55,
            'stop_distance': 0.4, 'camera_to_front': 0.0,
            'max_linear': 0.08, 'max_angular': 0.30, 'confirmations': 3,
        }
        self.config = {}
        for key, value in defaults.items():
            self.declare_parameter(key, value, ParameterDescriptor(read_only=True))
            self.config[key] = self.get_parameter(key).value
        for key in ('frame_timeout', 'sync_slop', 'connected_timeout'):
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.config['camera_to_front']):
            raise ValueError('camera_to_front must be finite')
        if not 0.0 < self.config['confidence'] <= 1.0:
            raise ValueError('confidence must be in (0, 1]')
        for key, ceiling in (('max_linear', 0.20), ('max_angular', 0.60)):
            if not math.isfinite(self.config[key]) or not 0 < self.config[key] <= ceiling:
                raise ValueError(f'{key} must be positive and at most {ceiling}')
        self.policy = ApproachPolicy(
            stop_distance=self.config['stop_distance'], max_linear=self.config['max_linear'],
            max_angular=self.config['max_angular'], target_timeout=self.config['frame_timeout'],
            confirmations=self.config['confirmations'],
        )
        self.detector = detector or BottleDetector(
            self.config['model_path'], confidence=self.config['confidence'])
        self.armed = self.config['enable_motion']
        self.disarmed_reason = None
        self.state_reason = 'WAITING_IMAGES'
        self.frames_processed = 0
        self.frames_rejected = 0
        self.inference_seconds = None
        self.completed_frame_age = None
        self.result_interval = None
        self.last_completion_time = None
        self.last_rejection_reason = None
        self.last_target = None
        self.last_result_received = None
        self.last_result_stamp = None
        self.latest_color = None
        self.latest_depth = None
        self.camera_info = None
        self.last_color_stamp = None
        self.last_depth_stamp = None
        self.last_pair = None
        self.base_connected = False
        self.base_was_ready = False
        self.base_received = None
        self.epoch = 0
        self.lock = threading.Lock()
        self.pending = None
        self.result = None
        self.wake = threading.Event()
        self.exiting = threading.Event()
        self.worker_error = None
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
        self.command_pub = self.create_publisher(Twist, self.config['cmd_topic'], 1)
        self.status_pub = self.create_publisher(String, '~/status', 1)
        self.debug_pub = self.create_publisher(Image, '~/image', sensor_qos)
        self.create_subscription(Image, self.config['color_topic'], self.on_color, sensor_qos)
        self.create_subscription(Image, self.config['depth_topic'], self.on_depth, sensor_qos)
        self.create_subscription(CameraInfo, self.config['camera_info_topic'],
                                 self.on_info, sensor_qos)
        self.create_subscription(Bool, self.config['connected_topic'], self.on_connected, 1)
        self.create_service(Trigger, '~/start', self.start_service)
        self.create_service(Trigger, '~/stop', self.stop_service)
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.05, self.tick, clock=self.steady_clock)
        self.create_timer(0.5, self.publish_status, clock=self.steady_clock)
        self.worker = threading.Thread(target=self.infer_loop, name='bottle-inference', daemon=True)
        self.worker.start()
        self.get_logger().info(
            'Bottle detection started. Motion ' + ('enabled' if self.armed else 'disabled')
            + '; stop clearance ' + str(self.config['stop_distance']) + ' m.')

    def capture_fresh(self, stamp):
        now = self.get_clock().now().nanoseconds * 1e-9
        return math.isfinite(stamp) and stamp > 0 and -0.05 <= now - stamp <= self.config['frame_timeout']

    def invalidate(self, reason):
        self.policy.update(None, time.monotonic())
        self.last_target = None
        self.state_reason = reason
        self.last_rejection_reason = reason
        # Cancel in-flight work for an input outage; only a newly paired frame can recover.
        self.epoch += 1
        with self.lock:
            self.pending = None
            self.result = None

    def on_color(self, message):
        stamp = stamp_seconds(message)
        if not self.capture_fresh(stamp) or (self.last_color_stamp is not None
                                            and stamp <= self.last_color_stamp):
            self.latest_color = None
            self.invalidate('INVALID_COLOR_TIMESTAMP')
            return
        self.last_color_stamp = stamp
        self.latest_color = (message, time.monotonic())
        self.try_pair()

    def on_depth(self, message):
        stamp = stamp_seconds(message)
        if not self.capture_fresh(stamp) or (self.last_depth_stamp is not None
                                            and stamp <= self.last_depth_stamp):
            self.latest_depth = None
            self.invalidate('INVALID_DEPTH_TIMESTAMP')
            return
        self.last_depth_stamp = stamp
        self.latest_depth = (message, time.monotonic())
        self.try_pair()

    def on_info(self, message):
        if (message.width <= 0 or message.height <= 0
                or not all(math.isfinite(v) for v in message.k)
                or message.k[0] <= 0 or message.k[4] <= 0
                or not 0 <= message.k[2] < message.width
                or not 0 <= message.k[5] < message.height):
            self.camera_info = None
            self.invalidate('INVALID_CAMERA_INFO')
            return
        self.camera_info = message
        self.try_pair()

    def on_connected(self, message):
        self.base_connected = message.data
        self.base_received = time.monotonic()
        if message.data:
            self.base_was_ready = True
        if self.config['require_base_connected'] and not message.data:
            if self.base_was_ready:
                self.armed = False
                self.disarmed_reason = 'BASE_DISCONNECTED'
                self.invalidate('BASE_DISCONNECTED')
            else:
                self.state_reason = 'WAITING_BASE'
            self.publish_command(0.0, 0.0)

    def base_fresh(self, now):
        return (not self.config['require_base_connected']
                or (self.base_connected and self.base_received is not None
                    and 0 <= now - self.base_received < self.config['connected_timeout']))

    def try_pair(self):
        if self.latest_color is None or self.latest_depth is None or self.camera_info is None:
            return
        color, color_received = self.latest_color
        depth, depth_received = self.latest_depth
        color_stamp, depth_stamp = stamp_seconds(color), stamp_seconds(depth)
        if abs(color_stamp - depth_stamp) > self.config['sync_slop']:
            return
        pair = (color_stamp, depth_stamp)
        # Each confirmation needs a distinct color AND depth observation. Do
        # not apply old depth to a new color image while its depth is arriving.
        if self.last_pair is not None and (pair[0] <= self.last_pair[0]
                                          or pair[1] <= self.last_pair[1]):
            return
        if (not self.capture_fresh(color_stamp) or not self.capture_fresh(depth_stamp)
                or time.monotonic() - min(color_received, depth_received) >= self.config['frame_timeout']):
            self.invalidate('STALE_IMAGES')
            return
        info = self.camera_info
        if (color.width != depth.width or color.height != depth.height
                or info.width != color.width or info.height != color.height
                or not color.header.frame_id
                or color.header.frame_id != depth.header.frame_id
                or info.header.frame_id != color.header.frame_id):
            self.invalidate('DEPTH_NOT_ALIGNED_TO_COLOR')
            return
        self.last_pair = pair
        with self.lock:
            # Replace pending work instead of queueing images behind slow inference.
            self.pending = (self.epoch, color, depth, info,
                            min(color_received, depth_received), min(pair))
        self.wake.set()

    def infer_loop(self):
        while not self.exiting.is_set():
            self.wake.wait(timeout=0.1)
            self.wake.clear()
            with self.lock:
                item, self.pending = self.pending, None
            if item is None:
                continue
            epoch, color, depth, info, received, stamp = item
            if time.monotonic() - received >= self.config['frame_timeout']:
                continue
            started = time.monotonic()
            try:
                bgr, metres = bgr_image(color), depth_metres(depth)
                detections = self.detector.detect(bgr)
                targets = []
                for detection in detections:
                    if detection.confidence < self.config['confidence']:
                        continue
                    target = estimate_target(metres, detection.bbox, info.k[0],
                                             info.k[2], detection.confidence)
                    if target is not None:
                        clearance = target.distance - self.config['camera_to_front']
                        # A target already at/past the bumper must also cause a stop.
                        targets.append(replace(target, distance=max(0.001, clearance)))
                # Deterministic nearest candidate; policy confirms any target change.
                target = min(targets, key=lambda entry: entry.distance) if targets else None
                debug = bgr.copy()
                for detection in detections:
                    x1, y1, x2, y2 = map(int, detection.bbox)
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 220, 220), 2)
                if target is not None:
                    cv2.putText(debug, f'bottle {target.distance:.2f}m', (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(debug, 'No bottle with valid depth', (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
                result = (epoch, target, received, stamp, color.header, debug, None)
            except Exception as error:
                # An inference/image failure stops through the independent control timer.
                result = (epoch, None, received, stamp, color.header, None, str(error))
            completed = time.monotonic()
            result += (completed - started, completed)
            with self.lock:
                self.result = result

    def observation_fresh(self, now):
        return (self.last_result_received is not None
                and 0 <= now - self.last_result_received < self.config['frame_timeout']
                and self.capture_fresh(self.last_result_stamp))

    def tick(self):
        now = time.monotonic()
        with self.lock:
            result, self.result = self.result, None
        if result is not None:
            epoch, target, received, stamp, header, debug, error, duration, completed = result
            if epoch == self.epoch:
                self.inference_seconds = duration
                self.completed_frame_age = completed - received
                if self.last_completion_time is not None:
                    self.result_interval = completed - self.last_completion_time
                self.last_completion_time = completed
            if (epoch == self.epoch and now - received < self.config['frame_timeout']
                    and self.capture_fresh(stamp)):
                self.frames_processed += 1
                self.worker_error = error
                self.last_rejection_reason = 'INFERENCE_ERROR' if error else None
                self.last_result_received, self.last_result_stamp = received, stamp
                self.last_target = target
                self.policy.update(target, received)
                self.state_reason = 'INFERENCE_ERROR' if error else self.policy.state
                if debug is not None:
                    image = Image()
                    image.header = header
                    image.height, image.width = debug.shape[:2]
                    image.encoding = 'bgr8'
                    image.step = image.width * 3
                    # The ROS setter validates bytes pixel by pixel in Python.
                    # An unsigned-byte array takes its fast path, keeping large
                    # preview images from blocking the 20 Hz command timer.
                    image.data = array('B', debug.tobytes())
                    self.debug_pub.publish(image)
            elif epoch == self.epoch:
                self.frames_rejected += 1
                self.last_rejection_reason = 'INFERENCE_TOO_SLOW_OR_STALE'
                self.policy.update(None, now)
                self.state_reason = 'INFERENCE_TOO_SLOW_OR_STALE'
        if self.count_publishers(self.config['cmd_topic']) > 1:
            self.armed = False
            self.disarmed_reason = 'OTHER_COMMAND_PUBLISHER'
            self.state_reason = 'OTHER_COMMAND_PUBLISHER'
        if self.armed and not self.base_fresh(now):
            # Initial telemetry wait is normal; a confirmed link loss requires rearming.
            if self.base_was_ready:
                self.armed = False
                self.disarmed_reason = 'WAITING_BASE'
            self.state_reason = 'WAITING_BASE'
        fresh = self.observation_fresh(now)
        if not fresh:
            if self.state_reason not in ('OTHER_COMMAND_PUBLISHER', 'BASE_DISCONNECTED', 'WAITING_BASE'):
                self.state_reason = 'WAITING_FRESH_DETECTION'
        # A slow worker can leave short zero-command intervals between valid
        # results. Keep confirmation history: each next update independently
        # checks the original receive-time gap and target association. A missing
        # or invalid detection still resets immediately. Never move using an
        # expired observation, even when confirmation history is retained.
        linear, angular = self.policy.command(now, preserve_confirmations=True)
        if not fresh:
            linear, angular = 0.0, 0.0
        if self.armed and self.policy.state == 'REACHED':
            self.state_reason = 'REACHED'
        if not self.armed or not self.base_fresh(now):
            linear, angular = 0.0, 0.0
        if not self.armed and self.disarmed_reason is not None and self.config['enable_motion']:
            self.state_reason = self.disarmed_reason
        self.publish_command(linear, angular)

    def publish_command(self, linear, angular):
        if rclpy.ok(context=self.context):
            message = Twist()
            message.linear.x, message.angular.z = linear, angular
            self.command_pub.publish(message)

    def publish_status(self):
        now = time.monotonic()
        target = self.last_target
        result_age = (now - self.last_result_received
                      if self.last_result_received is not None else None)
        capture_age = (self.get_clock().now().nanoseconds * 1e-9 - self.last_result_stamp
                       if self.last_result_stamp is not None else None)
        message = String()
        message.data = json.dumps({
            'state': self.state_reason, 'armed': self.armed,
            'motion_allowed': self.config['enable_motion'],
            'frames_processed': self.frames_processed,
            'frames_rejected': self.frames_rejected,
            'confirmation_count': self.policy.confirmation_count,
            'result_fresh': self.observation_fresh(now),
            'result_age_s': result_age,
            'capture_age_s': capture_age,
            'inference_s': self.inference_seconds,
            'completed_frame_age_s': self.completed_frame_age,
            'result_interval_s': self.result_interval,
            'last_rejection': self.last_rejection_reason,
            'base_connected': self.base_connected,
            'base_ready': self.base_fresh(now),
            'base_status_age_s': now - self.base_received if self.base_received is not None else None,
            'distance_m': target.distance if target else None,
            'bearing_rad': target.bearing if target else None,
            'confidence': target.confidence if target else None,
            'error': self.worker_error,
        })
        self.status_pub.publish(message)

    def start_service(self, _request, response):
        if not self.config['enable_motion']:
            response.success = False
            response.message = 'Detection-only launch: restart with enable_motion:=true'
        elif not self.base_fresh(time.monotonic()):
            response.success = False
            response.message = 'No fresh base connection'
        elif self.count_publishers(self.config['cmd_topic']) > 1:
            response.success = False
            response.message = 'Another node publishes commands on the same topic'
        else:
            self.invalidate('WAITING_FRESH_DETECTION')
            self.policy.reset()
            self.armed = True
            self.disarmed_reason = None
            response.success = True
            response.message = 'New approach armed; waiting for fresh confirmed detections'
        return response

    def stop_service(self, _request, response):
        self.armed = False
        self.disarmed_reason = 'STOPPED'
        self.invalidate('STOPPED')
        self.publish_command(0.0, 0.0)
        response.success = True
        response.message = 'Approach stopped; call start to begin a new approach'
        return response

    def stop(self):
        self.armed = False
        self.exiting.set()
        self.wake.set()
        for _ in range(3):
            self.publish_command(0.0, 0.0)
            time.sleep(0.02)
        self.worker.join(timeout=0.2)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    stopped = False

    def handle_signal(*_args):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        node = BottleApproach()
        while rclpy.ok() and not stopped:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
