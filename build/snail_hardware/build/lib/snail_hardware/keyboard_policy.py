"""Pure keyboard state for a low-speed two-wheel chassis checkout."""

import math


MAX_LINEAR_SPEED = 0.20
MAX_ANGULAR_SPEED = 0.60
MAX_KEY_TIMEOUT = 1.0


def positive_limit(value, name, upper):
    """Reject unsafe configuration instead of silently changing its meaning."""
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number in (0, {upper}]')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number in (0, {upper}]') from exc
    if not math.isfinite(value) or not 0.0 < value <= upper:
        raise ValueError(f'{name} must be a finite number in (0, {upper}]')
    return value


class HeldKeyDrive:
    """Combine held keys; stop keys inhibit motion until they are released."""

    MOTION_KEYS = frozenset({'i', 'comma', 'j', 'l', 'u', 'o', 'm', 'period'})
    STOP_KEYS = frozenset({'k', 'space'})
    ALIASES = {
        'w': 'i', 's': 'comma', 'a': 'j', 'd': 'l',
        'up': 'i', 'down': 'comma', 'left': 'j', 'right': 'l',
        ',': 'comma', '.': 'period', ' ': 'space',
        'less': 'comma', 'greater': 'period',
    }

    def __init__(self, speed=0.10, turn=0.30):
        self.speed = positive_limit(speed, 'speed', MAX_LINEAR_SPEED)
        self.turn = positive_limit(turn, 'turn', MAX_ANGULAR_SPEED)
        self.held = set()
        self.stops = set()

    @classmethod
    def normalize(cls, keysym):
        key = keysym.lower()
        return cls.ALIASES.get(key, key)

    def press(self, keysym):
        key = self.normalize(keysym)
        if key in self.STOP_KEYS:
            self.held.clear()
            self.stops.add(key)
        elif key in self.MOTION_KEYS and not self.stops:
            self.held.add(key)

    def release(self, keysym):
        key = self.normalize(keysym)
        self.held.discard(key)
        self.stops.discard(key)

    def clear(self):
        self.held.clear()
        self.stops.clear()

    def velocity(self):
        if self.stops:
            return 0.0, 0.0
        forward = bool(self.held & {'i', 'u', 'o'})
        backward = bool(self.held & {'comma', 'm', 'period'})
        left = bool(self.held & {'j', 'u', 'm'})
        right = bool(self.held & {'l', 'o', 'period'})
        return (
            self.speed * (int(forward) - int(backward)),
            self.turn * (int(left) - int(right)),
        )


class PulseKeyDrive:
    """Terminal keypresses expire; terminal input has no release events."""

    def __init__(self, speed=0.10, turn=0.30, key_timeout=0.35):
        self.drive = HeldKeyDrive(speed, turn)
        self.key_timeout = positive_limit(key_timeout, 'key_timeout', MAX_KEY_TIMEOUT)
        self.last_motion_at = None

    def press(self, keysym, now):
        key = self.drive.normalize(keysym)
        if key in self.drive.STOP_KEYS:
            self.clear()
        elif key in self.drive.MOTION_KEYS:
            if not math.isfinite(now):
                self.clear()
                raise ValueError('keypress time must be finite')
            self.drive.clear()
            self.drive.press(key)
            self.last_motion_at = now

    def clear(self):
        self.drive.clear()
        self.last_motion_at = None

    def velocity(self, now):
        if self.last_motion_at is None:
            return 0.0, 0.0
        age = now - self.last_motion_at
        if not math.isfinite(age) or age < 0.0 or age >= self.key_timeout:
            self.clear()
            return 0.0, 0.0
        return self.drive.velocity()


class TerminalKeys:
    """Decode fragmented arrow sequences without turning their A/B/C/D bytes."""

    ARROWS = {'A': 'i', 'B': 'comma', 'C': 'l', 'D': 'j'}

    def __init__(self):
        self.escape = ''

    def feed(self, text):
        keys = []
        for char in text:
            if char == '\x1b':
                self.escape = char
            elif self.escape:
                self.escape += char
                if self.escape == '\x1b[':
                    continue
                if len(self.escape) == 3 and self.escape.startswith('\x1b['):
                    if char in self.ARROWS:
                        keys.append(self.ARROWS[char])
                        self.escape = ''
                    elif char.isdigit() or char == ';':
                        continue
                    else:
                        self.escape = ''
                elif self.escape.startswith('\x1b['):
                    # Discard modified/function-key sequences through their final byte.
                    if '@' <= char <= '~':
                        self.escape = ''
                else:
                    self.escape = ''
            elif char.isprintable():
                keys.append(char)
        return keys
