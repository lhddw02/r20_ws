"""Zero-motion serial probe: test feedback and request the model-check beep."""

import argparse
import math
import time

import serial

from snail_hardware.protocol import (
    PacketParser, TELEMETRY_ID, decode_telemetry, encode_robot_type, encode_velocity,
)
from snail_hardware.serial_io import open_port, write_packet


def probe(port, duration, robot_type):
    parser = PacketParser()
    count = 0
    latest = None
    last_feedback = None
    model_sent = False
    start = time.monotonic()
    next_send = start
    stop = encode_velocity(0.0, 0.0, 0.0)
    port.reset_input_buffer()
    try:
        while time.monotonic() - start < duration:
            now = time.monotonic()
            if now >= next_send:
                write_packet(port, stop)
                next_send = now + 0.05
            for packet_id, payload in parser.feed(port.read(min(port.in_waiting, 4096))):
                if packet_id == TELEMETRY_ID and len(payload) == 20:
                    latest = decode_telemetry(payload)
                    count += 1
                    last_feedback = now
                    if not model_sent:
                        write_packet(port, encode_robot_type(robot_type))
                        print('Model check sent: short beep = match; long beep = mismatch.')
                        model_sent = True
            time.sleep(0.005)
    finally:
        for _ in range(3):
            write_packet(port, stop)
            time.sleep(0.02)
    if latest is None or time.monotonic() - last_feedback >= 0.5:
        print(f'FAIL: no fresh valid telemetry (received {count} packets).')
        return 1
    print(f'PASS: {count} valid telemetry packets; latest battery '
          f'{latest.battery_voltage:.2f} V; velocity {latest.velocity}.')
    print('This confirms STM32-to-Jetson feedback and host writes. '
          'The observed beep confirms STM32 command reception and model match; '
          'telemetry alone is not a command acknowledgement.')
    return 0


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--baud', type=int, default=230400)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--robot-type', default='r20_twd', choices=['r20_twd'])
    values = parser.parse_args(args)
    if not math.isfinite(values.duration) or values.duration <= 0 or values.baud <= 0:
        parser.error('duration and baud must be positive')
    try:
        with open_port(values.port, values.baud) as port:
            return probe(port, values.duration, values.robot_type)
    except (OSError, serial.SerialException) as error:
        print(f'FAIL: {error}')
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
