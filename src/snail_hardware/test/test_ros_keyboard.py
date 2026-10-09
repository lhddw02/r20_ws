"""Drive the real Tk bindings and observe real ROS Twist messages under Xvfb."""

import os
import time
import unittest
import uuid

if os.name != 'posix':
    raise unittest.SkipTest('ROS keyboard integration requires a Linux/POSIX display')

try:
    import pytest
except ImportError as error:
    raise unittest.SkipTest('ROS keyboard integration requires pytest') from error

rclpy = pytest.importorskip('rclpy', reason='ROS 2 is not installed')
tk = pytest.importorskip('tkinter', reason='Tk is not installed')

from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor

from snail_hardware.keyboard import KeyboardControl


@pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='Requires DISPLAY; run with xvfb-run')
def test_gui_keys_release_focus_loss_and_quit_publish_real_ros_stops(monkeypatch):
    """No mocked ROS publisher: Tk events must reach a separate topic subscriber."""
    namespace = '/r20_keyboard_test_' + uuid.uuid4().hex[:10]
    control = None
    peer = None
    executor = None
    window = None
    initialized = False
    observed = []
    errors = []
    completed = []
    real_tk = tk.Tk
    stage = 'discover'
    stage_started = time.monotonic()
    mark = 0

    def moving_since_mark():
        return any(abs(linear - 0.10) < 1e-6 and angular == 0.0
                   for linear, angular in observed[mark:])

    def stopped_since_mark():
        return (0.0, 0.0) in observed[mark:]

    def event(sequence, **options):
        window.event_generate(sequence, **options)

    def advance():
        nonlocal stage, stage_started, mark
        try:
            # The GUI spins its own node; this separate executor only spins the peer.
            executor.spin_once(timeout_sec=0.0)
            assert time.monotonic() - stage_started < 3.0, f'Timed out at GUI stage {stage}'
            if stage == 'discover':
                if (control.publisher.get_subscription_count() < 1
                        or peer.count_publishers('cmd_vel') < 1):
                    window.after(10, advance)
                    return
                window.focus_force()
                mark = len(observed)
                stage = 'press'
                stage_started = time.monotonic()
                window.after(30, lambda: event('<KeyPress>', keysym='w'))
            elif stage == 'press' and moving_since_mark():
                mark = len(observed)
                event('<KeyRelease>', keysym='w')
                stage = 'release'
                stage_started = time.monotonic()
            elif stage == 'release' and stopped_since_mark():
                completed.append('release')
                mark = len(observed)
                event('<KeyPress>', keysym='w')
                stage = 'press_before_focus_loss'
                stage_started = time.monotonic()
            elif stage == 'press_before_focus_loss' and moving_since_mark():
                mark = len(observed)
                event('<FocusOut>')
                stage = 'focus_loss'
                stage_started = time.monotonic()
            elif stage == 'focus_loss' and stopped_since_mark():
                completed.append('focus_loss')
                mark = len(observed)
                event('<KeyPress>', keysym='w')
                stage = 'press_before_quit'
                stage_started = time.monotonic()
            elif stage == 'press_before_quit' and moving_since_mark():
                mark = len(observed)
                completed.append('quit_while_moving')
                event('<KeyPress>', keysym='q')
                return
            window.after(10, advance)
        except BaseException as error:
            # Tk normally prints callback exceptions and keeps mainloop alive.
            # Preserve the failure and terminate the GUI so pytest can report it.
            errors.append(error)
            control.stop()
            window.destroy()

    def create_window(*args, **kwargs):
        nonlocal window
        window = real_tk(*args, **kwargs)
        window.after(100, advance)
        return window

    try:
        rclpy.init(args=['--ros-args', '-r', '__ns:=' + namespace])
        initialized = True
        control = KeyboardControl()
        peer = rclpy.create_node('keyboard_integration_peer')
        subscription = peer.create_subscription(
            Twist, 'cmd_vel',
            lambda message: observed.append((message.linear.x, message.angular.z)), 10,
        )
        assert subscription is not None
        executor = SingleThreadedExecutor()
        executor.add_node(peer)
        monkeypatch.setattr(tk, 'Tk', create_window)
        try:
            control.run_gui()
        except RuntimeError as error:
            if 'needs a display' in str(error):
                pytest.skip(str(error))
            raise
        if errors:
            raise errors[0]
        assert completed == ['release', 'focus_loss', 'quit_while_moving']
        assert control.stop_requested.is_set()

        # run_gui publishes zero both in Q handling and before returning. Keep
        # ROS alive long enough to observe shutdown's zero through actual DDS.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not stopped_since_mark():
            executor.spin_once(timeout_sec=0.01)
        assert stopped_since_mark(), 'GUI quit failed to publish a ROS zero command'
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert observed[-1] == (0.0, 0.0)
    finally:
        if control is not None:
            control.stop()
        if executor is not None:
            executor.shutdown()
        if peer is not None:
            peer.destroy_node()
        if control is not None:
            control.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()
