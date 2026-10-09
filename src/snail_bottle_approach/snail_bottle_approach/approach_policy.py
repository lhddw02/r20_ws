"""A conservative, ROS-independent bottle approach state machine.

Coordinates follow the camera: a positive bearing points to image right. ROS
base_link yaw is positive to the left, so the commanded angular sign is reversed.
Distance is optical Z in metres, not clearance from the front of the chassis.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class Target:
    bbox: Tuple[float, float, float, float]
    distance: float
    bearing: float
    confidence: float


def _valid_target(target: Target) -> bool:
    try:
        x1, y1, x2, y2 = target.bbox
        values = (x1, y1, x2, y2, target.distance, target.bearing,
                  target.confidence)
        return (all(math.isfinite(value) for value in values)
                and x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1
                and target.distance > 0
                and abs(target.bearing) < math.pi / 2
                and 0 < target.confidence <= 1)
    except (AttributeError, TypeError, ValueError):
        return False


class ApproachPolicy:
    """Confirm a stable target, approach slowly, and latch at the stop distance.

    ``update`` must be called once for each new camera frame, with monotonic
    capture time or monotonic processing time. Reusing an identical timestamp
    cannot count as another confirmation. ``None`` immediately stops movement.
    A gap between observation timestamps or a target change requires confirmation
    again. By default an expired command observation also clears confirmations.
    A caller may preserve them across command expiry when slow processing leaves
    brief gaps between individually fresh results. This never permits a command
    from an expired observation. Bounding boxes provide approximate association;
    they cannot distinguish overlapping bottles.
    Within the slowdown zone, a small forward speed floor prevents millimetre-
    per-second serial quantization from rounding progress to zero. The floor is
    capped by max_linear and ends exactly at the measured stop threshold.
    """

    def __init__(self, stop_distance: float = 0.40,
                 max_linear: float = 0.08, max_angular: float = 0.30,
                 target_timeout: float = 0.35, confirmations: int = 3,
                 *, confirmation_iou: float = 0.20,
                 confirmation_center_fraction: float = 0.20,
                 alignment_threshold: float = 0.15,
                 slowdown_distance: float = 0.50,
                 angular_gain: float = 1.5,
                 min_linear: float = 0.01):
        positive = (stop_distance, max_linear, max_angular, target_timeout,
                    confirmation_center_fraction, alignment_threshold,
                    slowdown_distance, angular_gain, min_linear)
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError('Policy distances, speeds, gains and timeouts must be finite and positive')
        if (isinstance(confirmations, bool) or not isinstance(confirmations, int)
                or confirmations < 1):
            raise ValueError('confirmations must be a positive integer')
        if not math.isfinite(confirmation_iou) or not 0 < confirmation_iou <= 1:
            raise ValueError('confirmation_iou must be in (0, 1]')
        if alignment_threshold >= math.pi / 2:
            raise ValueError('alignment_threshold must be less than pi/2')
        self.stop_distance = stop_distance
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.target_timeout = target_timeout
        self.confirmations = confirmations
        self.confirmation_iou = confirmation_iou
        self.confirmation_center_fraction = confirmation_center_fraction
        self.alignment_threshold = alignment_threshold
        self.slowdown_distance = slowdown_distance
        self.angular_gain = angular_gain
        self.min_linear = min(min_linear, max_linear)
        self.reset()

    def reset(self) -> None:
        self.state = 'WAITING'
        self._target: Optional[Target] = None
        self._last_update: Optional[float] = None
        self._count = 0

    @property
    def confirmation_count(self) -> int:
        """Number of distinct, associated observations in the current history."""
        return self._count

    def _associated(self, first: Target, second: Target) -> bool:
        ax1, ay1, ax2, ay2 = first.bbox
        bx1, by1, bx2, by2 = second.bbox
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        intersection = (max(0.0, min(ax2, bx2) - max(ax1, bx1))
                        * max(0.0, min(ay2, by2) - max(ay1, by1)))
        iou = intersection / (area_a + area_b - intersection)
        center_shift = math.hypot((ax1 + ax2 - bx1 - bx2) / 2,
                                  (ay1 + ay2 - by1 - by2) / 2)
        smaller_diagonal = min(math.hypot(ax2 - ax1, ay2 - ay1),
                               math.hypot(bx2 - bx1, by2 - by1))
        return (iou >= self.confirmation_iou
                or center_shift <= self.confirmation_center_fraction * smaller_diagonal)

    def update(self, target: Optional[Target], now: float) -> None:
        if self.state == 'REACHED':
            return
        if not math.isfinite(now) or target is None or not _valid_target(target):
            self.reset()
            return
        if self._last_update is not None:
            if now < self._last_update:
                self.reset()
                return
            if now == self._last_update:
                return
        stable = (self._target is not None and self._last_update is not None
                  and now - self._last_update <= self.target_timeout
                  and self._associated(self._target, target))
        self._count = self._count + 1 if stable else 1
        self._target = target
        self._last_update = now
        if self._count < self.confirmations:
            self.state = 'CONFIRMING'
        elif target.distance <= self.stop_distance:
            self.state = 'REACHED'
        else:
            self.state = 'APPROACHING'

    def command(self, now: float, *, preserve_confirmations: bool = False) -> Tuple[float, float]:
        """Return motion only while the latest observation remains fresh.

        With ``preserve_confirmations=True``, expiry commands zero while retaining
        the history needed to compare the next observation's original timestamp.
        ``update`` still clears confirmation after a genuine observation gap,
        missing/invalid target, clock regression or target change. The caller must
        pass original observation times, never inference completion times.
        """
        if self.state == 'REACHED':
            return 0.0, 0.0
        if (not math.isfinite(now) or self._last_update is None
                or now < self._last_update):
            self.reset()
            return 0.0, 0.0
        if now - self._last_update >= self.target_timeout:
            if not preserve_confirmations:
                self.reset()
            return 0.0, 0.0
        if self.state != 'APPROACHING' or self._target is None:
            return 0.0, 0.0
        target = self._target
        if abs(target.bearing) > self.alignment_threshold:
            turn = max(-self.max_angular,
                       min(self.max_angular, -self.angular_gain * target.bearing))
            return 0.0, turn
        remaining = max(0.0, target.distance - self.stop_distance)
        if remaining == 0.0:
            return 0.0, 0.0
        speed = self.max_linear * min(1.0, remaining / self.slowdown_distance)
        speed = min(self.max_linear, max(self.min_linear, speed))
        return speed, 0.0
