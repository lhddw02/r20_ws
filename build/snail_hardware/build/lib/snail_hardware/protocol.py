"""R20 STM32 X-Protocol encoding and decoding, independent of ROS and serial.

Frames are ``AA 55 total_length frame_id payload checksum``. The checksum is
the sum of every preceding frame byte modulo 256. This layout and the telemetry
scales follow the supplied ``ax_uart2.c``, ``ax_uart3.c`` and ``ax_robot.c``.
"""

from dataclasses import dataclass
import math
from numbers import Real
import struct
from typing import List, Tuple


TELEMETRY_ID = 0x10
VELOCITY_ID = 0x50
ROBOT_TYPE_ID = 0x5A
HEADER = b"\xaa\x55"
# Firmware AX_UART*_SendPacket permits at most 32 payload bytes.
MAX_FRAME_LENGTH = 37
ROBOT_TYPES = {
    "r20_mec": 1,
    "r20_fwd": 2,
    "r20_akm": 3,
    "r20_twd": 4,
    "r20_tak": 5,
    "r20_omni": 6,
}
_KNOWN_LENGTHS = {TELEMETRY_ID: 25, VELOCITY_ID: 11, ROBOT_TYPE_ID: 6}
_TELEMETRY_FORMAT = struct.Struct(">9hH")
Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Telemetry:
    """One feedback sample in SI units, plus battery voltage in volts.

    Acceleration is m/s^2, gyro is rad/s, and velocity is (vx m/s, vy m/s,
    wz rad/s). The chassis IMU values are already mapped to robot axes by MCU.
    """

    acceleration: Vector3
    gyro: Vector3
    velocity: Vector3
    battery_voltage: float


def _encode(frame_id: int, payload: bytes) -> bytes:
    frame = HEADER + bytes((len(payload) + 5, frame_id)) + payload
    return frame + bytes((sum(frame) & 0xFF,))


def _velocity_integer(value: float) -> int:
    # Check the value before truncation: overflow must never wrap the signed
    # 16-bit field into a velocity in the opposite direction.
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not -32.768 <= value <= 32.767
        or not math.isfinite(value)
    ):
        raise ValueError("Velocity must be finite and between -32.768 and 32.767")
    # Match the vendor ROS driver's C++ cast (truncate toward zero).
    return int(value * 1000)


def encode_velocity(vx: float, vy: float, wz: float) -> bytes:
    """Encode body velocity in m/s, m/s, rad/s as the MCU's 0x50 frame."""
    payload = struct.pack(
        ">hhh", _velocity_integer(vx), _velocity_integer(vy), _velocity_integer(wz)
    )
    return _encode(VELOCITY_ID, payload)


def encode_robot_type(name: str) -> bytes:
    """Encode the vendor's exact robot type name as a 0x5A check request.

    Firmware compares this value with its compile-time ROBOT_TYPE and beeps;
    this request does not change the firmware's chassis kinematics.
    """
    if not isinstance(name, str) or name not in ROBOT_TYPES:
        raise ValueError("Robot type must be one of: " + ", ".join(ROBOT_TYPES))
    return _encode(ROBOT_TYPE_ID, bytes((ROBOT_TYPES[name],)))


def decode_telemetry(payload: bytes) -> Telemetry:
    """Decode the 20-byte 0x10 payload, rejecting other payload sizes."""
    if len(payload) != _TELEMETRY_FORMAT.size:
        raise ValueError("R20 telemetry payload must contain exactly 20 bytes")
    raw = _TELEMETRY_FORMAT.unpack(payload)
    acceleration_scale = 19.6 / 32768.0
    gyro_scale = math.radians(500.0) / 32768.0
    return Telemetry(
        acceleration=tuple(value * acceleration_scale for value in raw[0:3]),
        gyro=tuple(value * gyro_scale for value in raw[3:6]),
        velocity=tuple(value / 1000.0 for value in raw[6:9]),
        battery_voltage=raw[9] / 100.0,
    )


class PacketParser:
    """Incrementally recover checksum-valid frames from an arbitrary stream.

    Incomplete frames survive across calls. Bad headers, lengths and checksums
    are skipped. If a corrupt but plausible length precedes a complete valid
    frame, that later frame restores synchronization without waiting for the
    corrupt length. Known frame IDs also have their exact lengths checked.

    Pending data stays shorter than MAX_FRAME_LENGTH, and input is processed in
    small chunks so even a large noisy read cannot grow the internal buffer
    without bound. Unknown IDs may be returned; consumers can ignore them.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        """Number of pending stream bytes (always less than 37 after feed)."""
        return len(self._buffer)

    def reset(self) -> None:
        """Discard partial data, for example after reopening the serial port."""
        self._buffer.clear()

    def feed(self, data: bytes) -> List[Tuple[int, bytes]]:
        """Return all complete valid (frame_id, payload) pairs in this input."""
        packets = []
        # A chunk plus an unfinished candidate uses at most 73 buffer bytes.
        view = memoryview(data)
        for offset in range(0, len(view), MAX_FRAME_LENGTH):
            self._buffer.extend(view[offset : offset + MAX_FRAME_LENGTH])
            packets.extend(self._extract())
        return packets

    def _valid_length(self, start: int, length: int) -> bool:
        if not 5 <= length <= MAX_FRAME_LENGTH:
            return False
        if len(self._buffer) - start >= 4:
            expected = _KNOWN_LENGTHS.get(self._buffer[start + 3])
            return expected is None or length == expected
        return True

    def _later_complete_frame(self) -> int:
        """Find a checksum-valid restart following an incomplete candidate."""
        start = self._buffer.find(HEADER, 2)
        while start >= 0:
            available = len(self._buffer) - start
            if available >= 3:
                length = self._buffer[start + 2]
                if self._valid_length(start, length) and available >= length:
                    candidate = self._buffer[start : start + length]
                    if sum(candidate[:-1]) & 0xFF == candidate[-1]:
                        return start
            start = self._buffer.find(HEADER, start + 1)
        return -1

    def _extract(self) -> List[Tuple[int, bytes]]:
        packets = []
        while self._buffer:
            start = self._buffer.find(HEADER)
            if start < 0:
                # Retain an AA at the end, in case 55 arrives next time.
                last = self._buffer[-1]
                self._buffer.clear()
                if last == HEADER[0]:
                    self._buffer.append(last)
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 3:
                break
            length = self._buffer[2]
            if not self._valid_length(0, length):
                del self._buffer[0]
                continue
            if len(self._buffer) < length:
                restart = self._later_complete_frame()
                if restart < 0:
                    break
                del self._buffer[:restart]
                continue
            candidate = self._buffer[:length]
            if sum(candidate[:-1]) & 0xFF != candidate[-1]:
                del self._buffer[0]
                continue
            packets.append((candidate[3], bytes(candidate[4:-1])))
            del self._buffer[:length]
        return packets
