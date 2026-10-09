"""ROS 2 keyboard teleoperation with held GUI keys or expiring TTY pulses."""

import os
import select
import signal
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerOptions

from snail_hardware.keyboard_policy import HeldKeyDrive, PulseKeyDrive, TerminalKeys


class KeyboardControl(Node):
    PERIOD = 0.05

    def __init__(self):
        super().__init__('r20_keyboard')
        self.declare_parameter('mode', 'gui')
        self.declare_parameter('speed', 0.10)
        self.declare_parameter('turn', 0.30)
        self.declare_parameter('key_timeout', 0.35)
        self.mode = self.get_parameter('mode').value
        if self.mode not in {'gui', 'terminal'}:
            raise ValueError("mode must be 'gui' or 'terminal'")
        self.pulse = PulseKeyDrive(
            self.get_parameter('speed').value,
            self.get_parameter('turn').value,
            self.get_parameter('key_timeout').value,
        )
        self.drive = self.pulse.drive
        self.publisher = self.create_publisher(Twist, 'cmd_vel', QoSProfile(depth=1))
        self.stop_requested = threading.Event()

    def publish_velocity(self, linear=0.0, angular=0.0):
        if rclpy.ok(context=self.context):
            command = Twist()
            command.linear.x = linear
            command.angular.z = angular
            self.publisher.publish(command)

    def stop(self):
        self.drive.clear()
        self.pulse.clear()
        self.publish_velocity()

    def run_gui(self):
        # Keep terminal use independent of tkinter and display availability.
        try:
            import tkinter as tk
        except ImportError as exc:
            raise RuntimeError('GUI mode requires python3-tk; use terminal mode in a TTY') from exc
        try:
            window = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError('GUI mode needs a display; use terminal mode in a TTY') from exc
        pending_releases = {}
        status = tk.StringVar(value='Click this window, then hold a movement key')
        window.title('R20 TWD keyboard control')
        window.geometry('570x210')
        tk.Label(window, text='R20 • WHEEL BASE CHECKOUT', font=('', 15, 'bold')).pack(pady=12)
        tk.Label(window, text=(
            'Hold W / S to move, A / D to turn (arrows also work)\n'
            'I / , forward / reverse; J / L turn; U / O / M / . curve\n'
            'Release or leave this window to stop • Space / K stop • Q quit'
        ), justify='center').pack()
        tk.Label(window, textvariable=status, font=('', 11, 'bold')).pack(pady=12)

        def cancel_releases():
            for callback in pending_releases.values():
                window.after_cancel(callback)
            pending_releases.clear()

        def close():
            self.stop_requested.set()
            cancel_releases()
            self.stop()
            window.destroy()

        def press(event):
            key = self.drive.normalize(event.keysym)
            if key == 'q':
                close()
                return
            callback = pending_releases.pop(key, None)
            if callback is not None:
                window.after_cancel(callback)
            self.drive.press(key)
            if key in self.drive.MOTION_KEYS or key in self.drive.STOP_KEYS:
                self.publish_velocity(*self.drive.velocity())

        def finish_release(key):
            pending_releases.pop(key, None)
            self.drive.release(key)
            self.publish_velocity(*self.drive.velocity())

        def release(event):
            key = self.drive.normalize(event.keysym)
            if key in self.drive.MOTION_KEYS or key in self.drive.STOP_KEYS:
                callback = pending_releases.pop(key, None)
                if callback is not None:
                    window.after_cancel(callback)
                # X11 repeat emits release/press pairs. A repeat cancels this delay.
                pending_releases[key] = window.after(30, lambda: finish_release(key))

        def focus_lost(_event):
            cancel_releases()
            self.stop()

        def tick():
            if self.stop_requested.is_set() or not rclpy.ok(context=self.context):
                close()
                return
            rclpy.spin_once(self, timeout_sec=0.0)
            linear, angular = self.drive.velocity()
            self.publish_velocity(linear, angular)
            status.set(
                f'{linear:+.2f} m/s • {angular:+.2f} rad/s'
                if linear or angular else 'Stopped • Click this window to drive'
            )
            window.after(round(self.PERIOD * 1000), tick)

        window.bind('<KeyPress>', press)
        window.bind('<KeyRelease>', release)
        window.bind('<FocusOut>', focus_lost)
        window.protocol('WM_DELETE_WINDOW', close)
        window.after(0, tick)
        try:
            window.mainloop()
        finally:
            self.stop()

    def run_terminal(self):
        if not sys.stdin.isatty():
            raise RuntimeError(
                'Terminal mode requires its own interactive TTY. '
                'Run ros2 run snail_hardware r20_keyboard --ros-args -p mode:=terminal '
                '-r cmd_vel:=/r20/cmd_vel in another terminal.'
            )
        try:
            import termios
            import tty
        except ImportError as exc:
            raise RuntimeError('Terminal mode requires a Linux/POSIX TTY') from exc
        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        decoder = TerminalKeys()
        self.get_logger().info(
            'W/S move, A/D turn; I/, J/L U/O/M/. also work; Space/K stop; Q quit. '
            f'Terminal has no key-release events: tap/repeat keys; commands expire '
            f'after {self.pulse.key_timeout:.2f} s.'
        )
        next_tick = time.monotonic()
        try:
            tty.setcbreak(fd)
            while not self.stop_requested.is_set() and rclpy.ok(context=self.context):
                ready, _, _ = select.select(
                    [fd], [], [], max(0.0, min(self.PERIOD, next_tick - time.monotonic()))
                )
                if ready:
                    raw = os.read(fd, 64)
                    if not raw:
                        break
                    for key in decoder.feed(raw.decode('ascii', errors='ignore')):
                        key = HeldKeyDrive.normalize(key)
                        if key == 'q':
                            self.stop_requested.set()
                            self.stop()
                            break
                        self.pulse.press(key, time.monotonic())
                        if key in HeldKeyDrive.STOP_KEYS:
                            self.publish_velocity()
                now = time.monotonic()
                if now >= next_tick:
                    rclpy.spin_once(self, timeout_sec=0.0)
                    self.publish_velocity(*self.pulse.velocity(now))
                    next_tick = now + self.PERIOD
        finally:
            self.stop()
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    control = None
    previous_signals = {}
    result = 0
    try:
        control = KeyboardControl()
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.signal(
                signum, lambda *_: control.stop_requested.set()
            )
        if control.mode == 'gui':
            control.run_gui()
        else:
            control.run_terminal()
    except (ValueError, RuntimeError, OSError) as exc:
        if control is not None:
            control.get_logger().error(str(exc))
        else:
            print(f'R20 keyboard: {exc}', file=sys.stderr)
        result = 1
    finally:
        if control is not None:
            # Publish zero before destroying the node or the ROS context.
            control.stop()
            control.destroy_node()
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
        if rclpy.ok():
            rclpy.shutdown()
    return result


if __name__ == '__main__':
    sys.exit(main())
