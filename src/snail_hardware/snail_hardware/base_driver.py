"""R20 TWD Twist-to-STM32 bridge for ROS 2 Humble."""

import math
import signal
import time

import rclpy
import serial
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32

from snail_hardware.command_guard import CommandGuard
from snail_hardware.protocol import (
    PacketParser, TELEMETRY_ID, decode_telemetry, encode_robot_type, encode_velocity,
)
from snail_hardware.serial_io import open_port, write_packet


class BaseDriver(Node):
    def __init__(self):
        super().__init__('r20_base_driver')
        self.port = None
        self.failure = None
        self.stopping = False
        self.feedback_count = 0
        self.sent_count = 0
        self.model_check_sent = False
        self.warned_invalid = False
        defaults = {
            'port': '/dev/ttyUSB0', 'baud': 230400, 'robot_type': 'r20_twd',
            'command_timeout': 0.3, 'feedback_timeout': 0.5,
            'startup_timeout': 10.0, 'max_linear': 0.2, 'max_angular': 0.6,
            'enable_motion': True, 'imu_frame': 'r20_imu_link',
        }
        values = {}
        for key, value in defaults.items():
            self.declare_parameter(key, value, ParameterDescriptor(read_only=True))
            values[key] = self.get_parameter(key).value
        if values['robot_type'] != 'r20_twd':
            raise ValueError('This driver is configured for the R20 TWD prototype')
        if (not math.isfinite(values['startup_timeout'])
                or values['startup_timeout'] <= 0 or values['baud'] <= 0):
            raise ValueError('startup_timeout and baud must be positive')
        self.guard = CommandGuard(**{key: values[key] for key in (
            'command_timeout', 'feedback_timeout', 'max_linear', 'max_angular')})
        self.config = values
        self.parser = PacketParser()
        self.started = time.monotonic()
        self.last_rx_poll = self.started
        qos = QoSProfile(depth=1)
        self.velocity_pub = self.create_publisher(Twist, 'feedback_velocity', qos)
        self.imu_pub = self.create_publisher(Imu, 'imu/data_raw', qos)
        self.voltage_pub = self.create_publisher(Float32, 'battery_voltage', qos)
        self.connected_pub = self.create_publisher(Bool, 'connected', qos)
        self.diagnostic_pub = self.create_publisher(DiagnosticArray, 'diagnostics', qos)
        self.command_sub = self.create_subscription(Twist, 'cmd_vel', self.receive, qos)
        self.applied_pub = self.create_publisher(Twist, 'applied_cmd_vel', qos)
        try:
            self.port = open_port(values['port'], values['baud'])
            self.port.reset_input_buffer()
            self.send(encode_velocity(0.0, 0.0, 0.0))
        except Exception:
            if self.port is not None:
                self.port.close()
            raise
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.rx_timer = self.create_timer(0.01, self.read_feedback, clock=self.steady_clock)
        self.tx_timer = self.create_timer(0.05, self.send_command, clock=self.steady_clock)
        self.status_timer = self.create_timer(1.0, self.publish_status, clock=self.steady_clock)
        self.get_logger().info(
            f"Opened {values['port']} at {values['baud']} 8N1; waiting for STM32 telemetry")
        self.get_logger().warning(
            'The supplied STM32 firmware has no command watchdog. '
            'A host stop cannot protect against cable loss or Jetson failure.')
        if not values['enable_motion']:
            self.get_logger().info('Communication test mode: only zero velocity is transmitted')

    def send(self, packet):
        write_packet(self.port, packet)
        self.sent_count += 1

    def receive(self, message):
        components = (message.linear.x, message.linear.y, message.linear.z,
                      message.angular.x, message.angular.y, message.angular.z)
        if (not all(math.isfinite(v) for v in components)
                or any(v != 0 for v in components[1:5])):
            self.guard.clear()
            if not self.warned_invalid:
                self.get_logger().warning('Rejected invalid or non-planar TWD command; stopping')
                self.warned_invalid = True
            return
        self.warned_invalid = False
        if self.config['enable_motion']:
            self.guard.accept(message.linear.x, message.angular.z, time.monotonic())

    def read_feedback(self):
        if self.failure or self.stopping:
            return
        try:
            now = time.monotonic()
            poll_gap = now - self.last_rx_poll
            self.last_rx_poll = now
            if poll_gap >= self.guard.feedback_timeout:
                # After an executor stall, queued packets cannot establish liveness.
                self.port.reset_input_buffer()
                self.parser.reset()
                self.guard.last_feedback = None
                self.guard.clear()
                return
            count = self.port.in_waiting
            if count > 4096:
                # A backlog cannot be treated as evidence that telemetry is current.
                self.port.reset_input_buffer()
                self.parser.reset()
                self.guard.last_feedback = None
                self.guard.clear()
                return
            for packet_id, payload in self.parser.feed(self.port.read(count)):
                if packet_id != TELEMETRY_ID or len(payload) != 20:
                    continue
                data = decode_telemetry(payload)
                self.guard.feedback(time.monotonic())
                self.feedback_count += 1
                if not self.model_check_sent:
                    self.send(encode_robot_type(self.config['robot_type']))
                    self.model_check_sent = True
                    self.get_logger().info(
                        'Valid STM32 telemetry received. TWD model check sent: '
                        'short beep = match; long beep = mismatch. '
                        'This command does not change STM32 kinematics.')
                self.publish_feedback(data)
        except (OSError, serial.SerialException) as error:
            self.fail(error)

    def send_command(self):
        if self.failure or self.stopping:
            return
        now = time.monotonic()
        if self.feedback_count == 0 and now - self.started >= self.config['startup_timeout']:
            self.fail(RuntimeError('No valid STM32 telemetry before startup_timeout'))
            return
        linear, angular = self.guard.output(now)
        try:
            self.send(encode_velocity(linear, 0.0, angular))
            message = Twist()
            message.linear.x = linear
            message.angular.z = angular
            self.applied_pub.publish(message)
        except (OSError, serial.SerialException) as error:
            self.fail(error)

    def publish_feedback(self, data):
        velocity = Twist()
        velocity.linear.x, velocity.linear.y, velocity.angular.z = data.velocity
        self.velocity_pub.publish(velocity)
        imu = Imu()
        imu.header.stamp = self.get_clock().now().to_msg()
        imu.header.frame_id = self.config['imu_frame']
        imu.orientation.w = 1.0
        imu.orientation_covariance[0] = -1.0  # The packet contains no orientation.
        (imu.linear_acceleration.x, imu.linear_acceleration.y,
         imu.linear_acceleration.z) = data.acceleration
        (imu.angular_velocity.x, imu.angular_velocity.y,
         imu.angular_velocity.z) = data.gyro
        self.imu_pub.publish(imu)
        voltage = Float32()
        voltage.data = data.battery_voltage
        self.voltage_pub.publish(voltage)

    def publish_status(self):
        fresh = self.guard.feedback_fresh(time.monotonic()) and self.failure is None
        connected = Bool()
        connected.data = fresh
        self.connected_pub.publish(connected)
        status = DiagnosticStatus()
        status.name = 'r20/serial'
        status.hardware_id = self.config['port']
        status.level = DiagnosticStatus.OK if fresh else DiagnosticStatus.ERROR
        status.message = ('Telemetry receiving; model match requires observing the beep'
                          if fresh else 'No fresh telemetry; transmitting stop')
        if self.failure:
            status.message = str(self.failure)
        status.values = [KeyValue(key=key, value=str(value)) for key, value in {
            'feedback_packets': self.feedback_count, 'sent_packets': self.sent_count,
            'enable_motion': self.config['enable_motion'],
            'robot_type': self.config['robot_type'],
        }.items()]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostic_pub.publish(message)

    def fail(self, error):
        self.failure = error
        self.guard.clear()
        self.get_logger().error(f'Serial link failed: {error}; ending the driver')

    def stop(self):
        self.stopping = True
        self.guard.clear()
        if self.port is not None and self.port.is_open:
            try:
                # No unbounded flush: each write has a finite timeout.
                for _ in range(3):
                    self.send(encode_velocity(0.0, 0.0, 0.0))
                    time.sleep(0.02)
            except (OSError, serial.SerialException) as error:
                self.get_logger().error(f'Could not deliver final stop: {error}')
            finally:
                self.port.close()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    requested_stop = False

    def shutdown_signal(*_args):
        nonlocal requested_stop
        requested_stop = True

    signal.signal(signal.SIGINT, shutdown_signal)
    signal.signal(signal.SIGTERM, shutdown_signal)
    try:
        node = BaseDriver()
        while rclpy.ok() and not requested_stop and node.failure is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.failure:
            raise RuntimeError(str(node.failure))
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
