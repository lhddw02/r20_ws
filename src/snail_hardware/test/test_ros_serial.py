"""Real ROS 2 + pyserial integration against an independently simulated STM32.

These tests require Linux/POSIX, ROS 2 and pyserial. They do not require a
physical robot: a pseudo-terminal supplies the STM32 side of the serial link.
They deliberately avoid the production protocol encoder/parser when checking
wire bytes, so the serial bridge is tested across its actual I/O boundaries.
"""

import errno
import math
import os
import struct
import time
import unittest
import uuid

if os.name != "posix":
    raise unittest.SkipTest("ROS serial integration requires POSIX pseudo-terminals")

try:
    import pytest
except ImportError as error:
    raise unittest.SkipTest("ROS serial integration requires pytest") from error

rclpy = pytest.importorskip("rclpy", reason="ROS 2 is not installed")
pytest.importorskip("serial", reason="pyserial is not installed")

from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from snail_hardware.base_driver import BaseDriver


ZERO_COMMAND = bytes.fromhex("AA 55 0B 50 00 00 00 00 00 00 5A")
MOVING_COMMAND = bytes.fromhex("AA 55 0B 50 00 64 00 00 FE D4 90")
CLAMPED_COMMAND = bytes.fromhex("AA 55 0B 50 00 C8 00 00 FD A8 C7")
TWD_CHECK = bytes.fromhex("AA 55 06 5A 04 63")

# Constructed independently from the firmware's documented raw field layout.
_RAW_SAMPLE = (16384, -8192, 32767, 16384, -16384, 0, 100, -20, -300, 1234)
_TELEMETRY_BODY = bytes.fromhex("AA 55 19 10") + struct.pack(">9hH", *_RAW_SAMPLE)
TELEMETRY = _TELEMETRY_BODY + bytes((sum(_TELEMETRY_BODY) & 0xFF,))


def command(linear=0.1, angular=-0.3):
    message = Twist()
    message.linear.x = linear
    message.angular.z = angular
    return message


class FakeSTM32:
    """Raw PTY endpoint; validate every received frame independently."""

    def __init__(self):
        self.master, self.slave = os.openpty()
        self.port_name = os.ttyname(self.slave)
        os.set_blocking(self.master, False)
        self.pending = bytearray()
        self.frames = []

    def send_feedback(self, packet=TELEMETRY):
        assert os.write(self.master, packet) == len(packet)

    def read_commands(self):
        while True:
            try:
                chunk = os.read(self.master, 4096)
            except BlockingIOError:
                break
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            self.pending.extend(chunk)
        while len(self.pending) >= 3:
            assert self.pending[:2] == b"\xaa\x55", "Driver wrote an invalid frame header"
            length = self.pending[2]
            assert 5 <= length <= 37, "Driver wrote an invalid total frame length"
            if len(self.pending) < length:
                break
            packet = bytes(self.pending[:length])
            assert sum(packet[:-1]) & 0xFF == packet[-1], "Driver wrote a bad checksum"
            assert packet[3] in (0x50, 0x5A), "Unexpected command frame ID"
            assert length == (11 if packet[3] == 0x50 else 6)
            self.frames.append(packet)
            del self.pending[:length]

    def velocities_since(self, mark=0):
        self.read_commands()
        return [packet for packet in self.frames[mark:] if packet[3] == 0x50]

    def mark(self):
        self.read_commands()
        return len(self.frames)

    def close(self):
        os.close(self.master)
        os.close(self.slave)


class ROSSerialHarness:
    """Run actual driver and topic peers on one executor with monotonic bounds."""

    def __init__(self, enable_motion=True, startup_timeout=1.0):
        self.fake = FakeSTM32()
        self.driver = None
        self.peer = None
        self.executor = None
        self.initialized = False
        self.closed = False
        self.observed = {"velocity": [], "imu": [], "voltage": [], "applied": []}
        self.last_feedback_send = 0.0
        self.last_command_send = 0.0
        self.namespace = "/r20_test_" + uuid.uuid4().hex[:10]
        args = ["--ros-args", "-r", "__ns:=" + self.namespace]
        parameters = {
            "port": self.fake.port_name,
            "startup_timeout": startup_timeout,
            "feedback_timeout": 0.15,
            "command_timeout": 0.1,
            "max_linear": 0.2,
            "max_angular": 0.6,
            "enable_motion": "true" if enable_motion else "false",
        }
        for name, value in parameters.items():
            args.extend(["-p", f"{name}:={value}"])
        try:
            rclpy.init(args=args)
            self.initialized = True
            self.driver = BaseDriver()
            self.peer = rclpy.create_node("serial_integration_peer")
            self.command_pub = self.peer.create_publisher(Twist, "cmd_vel", 10)
            self.subscriptions = []
            for topic, message_type, key in (
                ("feedback_velocity", Twist, "velocity"),
                ("imu/data_raw", Imu, "imu"),
                ("battery_voltage", Float32, "voltage"),
                ("applied_cmd_vel", Twist, "applied"),
            ):
                self.subscriptions.append(self.peer.create_subscription(
                    message_type, topic, self.observed[key].append, 10))
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.driver)
            self.executor.add_node(self.peer)
            self.fake.read_commands()
        except Exception:
            self.close()
            raise

    def step(self, feedback=False, message=None):
        now = time.monotonic()
        if feedback and now - self.last_feedback_send >= 0.02:
            self.fake.send_feedback()
            self.last_feedback_send = now
        if message is not None and now - self.last_command_send >= 0.02:
            self.command_pub.publish(message)
            self.last_command_send = now
        self.executor.spin_once(timeout_sec=0.005)
        self.fake.read_commands()

    def run(self, duration, feedback=False, message=None):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.step(feedback=feedback, message=message)

    def until(self, predicate, feedback=False, message=None, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.step(feedback=feedback, message=message)
            if predicate():
                return
        assert predicate(), "Timed out while waiting for ROS/serial integration condition"

    def discover(self):
        # DDS graph updates happen asynchronously; wait for both directions.
        self.until(
            lambda: self.command_pub.get_subscription_count() >= 1
            and all(self.peer.count_publishers(topic) >= 1 for topic in (
                "feedback_velocity", "imu/data_raw", "battery_voltage", "applied_cmd_vel")),
            timeout=0.8,
        )
        assert self.driver.failure is None

    def stop_and_check_wire(self):
        mark = self.fake.mark()
        self.driver.stop()
        assert not self.driver.port.is_open
        assert self.fake.velocities_since(mark) == [ZERO_COMMAND] * 3

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.driver is not None:
                self.driver.stop()
                self.fake.read_commands()
            if self.executor is not None:
                self.executor.shutdown()
            if self.peer is not None:
                self.peer.destroy_node()
            if self.driver is not None:
                self.driver.destroy_node()
            if self.initialized and rclpy.ok():
                rclpy.shutdown()
        finally:
            self.fake.close()


@pytest.fixture
def driver_factory():
    harnesses = []

    def create(**options):
        harness = ROSSerialHarness(**options)
        harnesses.append(harness)
        return harness

    yield create
    for harness in reversed(harnesses):
        harness.close()


def test_bidirectional_ros_serial_motion_gates_and_shutdown(driver_factory):
    harness = driver_factory()
    harness.discover()

    # A topic command alone cannot start the MCU. A checksum-corrupt telemetry
    # packet must not count as feedback or trigger the chassis type check.
    harness.run(0.12, message=command())
    harness.fake.send_feedback(TELEMETRY[:-1] + bytes((TELEMETRY[-1] ^ 1,)))
    harness.run(0.04, message=command())
    assert harness.driver.feedback_count == 0
    assert harness.fake.velocities_since() and all(
        packet == ZERO_COMMAND for packet in harness.fake.velocities_since())
    assert TWD_CHECK not in harness.fake.frames

    mark = harness.fake.mark()
    harness.until(lambda: all(harness.observed.values()), feedback=True)
    assert harness.driver.failure is None
    assert harness.fake.frames.count(TWD_CHECK) == 1
    assert all(packet == ZERO_COMMAND for packet in harness.fake.velocities_since(mark))
    velocity = harness.observed["velocity"][-1]
    assert (velocity.linear.x, velocity.linear.y, velocity.angular.z) == pytest.approx(
        (0.1, -0.02, -0.3))
    imu = harness.observed["imu"][-1]
    assert imu.header.frame_id == "r20_imu_link"
    assert imu.header.stamp.sec > 0
    assert imu.orientation_covariance[0] == -1.0
    assert (imu.linear_acceleration.x, imu.linear_acceleration.y,
            imu.linear_acceleration.z) == pytest.approx(
                (9.8, -4.9, 19.6 * 32767 / 32768))
    assert (imu.angular_velocity.x, imu.angular_velocity.y,
            imu.angular_velocity.z) == pytest.approx(
                (250 * math.pi / 180, -250 * math.pi / 180, 0.0))
    assert harness.observed["voltage"][-1].data == pytest.approx(12.34, abs=1e-5)

    # The actual outgoing bytes must clamp both permitted TWD axes.
    mark = harness.fake.mark()
    harness.until(lambda: CLAMPED_COMMAND in harness.fake.velocities_since(mark),
                  feedback=True, message=command(1.0, -3.0))
    harness.until(lambda: any(abs(sample.linear.x - 0.2) < 1e-6
                              and abs(sample.angular.z + 0.6) < 1e-6
                              for sample in harness.observed["applied"]), feedback=True)

    for invalid in ("lateral", "nan", "infinity"):
        mark = harness.fake.mark()
        harness.until(lambda: MOVING_COMMAND in harness.fake.velocities_since(mark),
                      feedback=True, message=command())
        rejected = command()
        if invalid == "lateral":
            rejected.linear.y = 0.01
        elif invalid == "nan":
            rejected.linear.x = math.nan
        else:
            rejected.angular.z = math.inf
        harness.run(0.15, feedback=True, message=rejected)
        stopped = harness.fake.velocities_since()
        assert stopped[-2:] == [ZERO_COMMAND, ZERO_COMMAND], invalid

    # Keeping feedback alive cannot keep an expired command alive.
    mark = harness.fake.mark()
    harness.until(lambda: MOVING_COMMAND in harness.fake.velocities_since(mark),
                  feedback=True, message=command())
    mark = harness.fake.mark()
    harness.run(0.25, feedback=True)
    assert harness.fake.velocities_since(mark)[-2:] == [ZERO_COMMAND, ZERO_COMMAND]

    # Conversely, a stream of commands cannot keep motion alive after receive
    # feedback disappears. Do not resume the discarded command on recovery.
    mark = harness.fake.mark()
    harness.until(lambda: MOVING_COMMAND in harness.fake.velocities_since(mark),
                  feedback=True, message=command())
    mark = harness.fake.mark()
    harness.run(0.35, message=command())
    assert harness.fake.velocities_since(mark)[-3:] == [ZERO_COMMAND] * 3
    assert harness.driver.failure is None
    harness.run(0.08)  # Deliver any queued DDS command while feedback is absent.
    mark = harness.fake.mark()
    harness.run(0.15, feedback=True)
    recovered = harness.fake.velocities_since(mark)
    assert recovered and all(packet == ZERO_COMMAND for packet in recovered)
    mark = harness.fake.mark()
    harness.until(lambda: MOVING_COMMAND in harness.fake.velocities_since(mark),
                  feedback=True, message=command())

    # Simulate an executor stall without sleeping through DDS deadlines: the
    # UART buffer contains telemetry from before a deliberately old RX poll.
    # These buffered bytes cannot turn a dead receive link back into a ready one.
    feedback_count = harness.driver.feedback_count
    harness.fake.send_feedback()
    deadline = time.monotonic() + 0.1
    while harness.driver.port.in_waiting < len(TELEMETRY) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert harness.driver.port.in_waiting >= len(TELEMETRY)
    harness.driver.last_rx_poll = time.monotonic() - 1.0
    harness.driver.read_feedback()
    assert harness.driver.feedback_count == feedback_count
    assert not harness.driver.guard.feedback_fresh(time.monotonic())
    assert harness.driver.guard.command == (0.0, 0.0)
    assert harness.driver.port.in_waiting == 0
    assert harness.driver.parser.buffered_bytes == 0
    mark = harness.fake.mark()
    harness.driver.send_command()
    harness.run(0.08, message=command())
    harness.run(0.03)  # Drain commands received before post-stall feedback.
    assert harness.fake.velocities_since(mark) and all(
        packet == ZERO_COMMAND for packet in harness.fake.velocities_since(mark))
    mark = harness.fake.mark()
    harness.run(0.15, feedback=True)
    assert harness.driver.feedback_count > feedback_count
    assert harness.fake.velocities_since(mark) and all(
        packet == ZERO_COMMAND for packet in harness.fake.velocities_since(mark))
    mark = harness.fake.mark()
    harness.until(lambda: MOVING_COMMAND in harness.fake.velocities_since(mark),
                  feedback=True, message=command())
    harness.stop_and_check_wire()


def test_disabled_motion_still_receives_ros_feedback_and_only_sends_stops(driver_factory):
    harness = driver_factory(enable_motion=False)
    harness.discover()
    harness.until(lambda: all(harness.observed.values()),
                  feedback=True, message=command(1.0, -3.0))
    harness.run(0.15, feedback=True, message=command(1.0, -3.0))
    assert harness.driver.feedback_count > 0
    assert harness.driver.failure is None
    assert harness.fake.frames.count(TWD_CHECK) == 1
    assert harness.fake.velocities_since() and all(
        packet == ZERO_COMMAND for packet in harness.fake.velocities_since())
    assert all(sample.linear.x == 0.0 and sample.angular.z == 0.0
               for sample in harness.observed["applied"])
    harness.stop_and_check_wire()


def test_startup_without_valid_stm32_feedback_fails_and_stops(driver_factory):
    harness = driver_factory(startup_timeout=0.5)
    harness.fake.send_feedback(TELEMETRY[:-1] + bytes((TELEMETRY[-1] ^ 1,)))
    harness.until(lambda: harness.driver.failure is not None, timeout=1.5)
    assert "No valid STM32 telemetry" in str(harness.driver.failure)
    assert harness.driver.feedback_count == 0
    assert TWD_CHECK not in harness.fake.frames
    assert harness.fake.velocities_since() and all(
        packet == ZERO_COMMAND for packet in harness.fake.velocities_since())
    harness.stop_and_check_wire()
