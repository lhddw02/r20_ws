"""Monotonic command and telemetry watchdogs, independent of ROS."""

import math


class CommandGuard:
    def __init__(self, command_timeout=0.3, feedback_timeout=0.5,
                 max_linear=0.2, max_angular=0.6):
        for name, value in locals().copy().items():
            if name == 'self':
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if max_linear > 1.5 or max_angular > 6.28:
            raise ValueError('Limits exceed the R20 firmware velocity limits')
        self.command_timeout = command_timeout
        self.feedback_timeout = feedback_timeout
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.last_feedback = None
        self.clear()

    def clear(self):
        self.command = (0.0, 0.0)
        self.last_command = None

    def feedback_fresh(self, now):
        return (self.last_feedback is not None
                and 0 <= now - self.last_feedback < self.feedback_timeout)

    def feedback(self, now):
        # Do not replay a command that preceded a receive outage.
        if not self.feedback_fresh(now):
            self.clear()
        self.last_feedback = now

    def accept(self, linear, angular, now):
        if not all(math.isfinite(v) for v in (linear, angular, now)):
            self.clear()
            return False
        if not self.feedback_fresh(now):
            self.clear()
            return False
        self.command = (
            max(-self.max_linear, min(self.max_linear, linear)),
            max(-self.max_angular, min(self.max_angular, angular)),
        )
        self.last_command = now
        return True

    def output(self, now):
        if (not self.feedback_fresh(now) or self.last_command is None
                or not 0 <= now - self.last_command < self.command_timeout):
            self.clear()
        return self.command
