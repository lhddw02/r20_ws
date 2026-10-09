"""Offline tests for the byte protocol used by the supplied R20 firmware."""

from dataclasses import FrozenInstanceError
import math
import unittest

from snail_hardware.protocol import (
    MAX_FRAME_LENGTH,
    PacketParser,
    ROBOT_TYPES,
    decode_telemetry,
    encode_robot_type,
    encode_velocity,
)


# Hardcoded wire bytes independently derived from the firmware layout.
ZERO_COMMAND = bytes.fromhex("AA 55 0B 50 00 00 00 00 00 00 5A")
MOVING_COMMAND = bytes.fromhex("AA 55 0B 50 00 64 00 00 FE D4 90")
TWD_CHECK = bytes.fromhex("AA 55 06 5A 04 63")
TELEMETRY_FRAME = bytes.fromhex(
    "AA 55 19 10 80 00 7F FF FF FF 80 00 7F FF 00 00 "
    "00 64 FE D4 80 00 FF FF D6"
)


class EncodingTests(unittest.TestCase):
    def test_golden_velocity_commands(self):
        self.assertEqual(encode_velocity(0.0, 0.0, 0.0), ZERO_COMMAND)
        self.assertEqual(encode_velocity(0.1, 0.0, -0.3), MOVING_COMMAND)

    def test_golden_twd_check(self):
        self.assertEqual(encode_robot_type("r20_twd"), TWD_CHECK)
        for name, expected_id in ROBOT_TYPES.items():
            self.assertEqual(encode_robot_type(name)[4], expected_id)

    def test_robot_type_requires_exact_known_name(self):
        for bad_name in ("twd", "R20_TWD", "r20_twd4", "", 4, None):
            with self.subTest(name=bad_name), self.assertRaises(ValueError):
                encode_robot_type(bad_name)

    def test_signed_range_and_truncation(self):
        frame = encode_velocity(-32.768, 32.767, -0.0009)
        self.assertEqual(frame[4:10], bytes.fromhex("80 00 7F FF 00 00"))
        self.assertEqual(encode_velocity(0.0019, -0.0019, 0.0)[4:8], b"\x00\x01\xff\xff")

    def test_nonfinite_and_out_of_range_velocities_are_rejected(self):
        invalid = (math.nan, math.inf, -math.inf, 32.768, -32.769, 1e100, "0.1", True)
        for value in invalid:
            for index in range(3):
                values = [0.0, 0.0, 0.0]
                values[index] = value
                with self.subTest(value=value, index=index), self.assertRaises(ValueError):
                    encode_velocity(*values)


class TelemetryTests(unittest.TestCase):
    def test_signed_extremes_and_unsigned_battery(self):
        telemetry = decode_telemetry(TELEMETRY_FRAME[4:-1])
        self.assertAlmostEqual(telemetry.acceleration[0], -19.6)
        self.assertAlmostEqual(telemetry.acceleration[1], 19.6 * 32767 / 32768)
        self.assertAlmostEqual(telemetry.acceleration[2], -19.6 / 32768)
        self.assertAlmostEqual(telemetry.gyro[0], -500 * math.pi / 180)
        self.assertAlmostEqual(telemetry.gyro[1], 500 * math.pi / 180 * 32767 / 32768)
        self.assertEqual(telemetry.gyro[2], 0.0)
        self.assertEqual(telemetry.velocity, (0.1, -0.3, -32.768))
        self.assertEqual(telemetry.battery_voltage, 655.35)

    def test_telemetry_is_immutable(self):
        telemetry = decode_telemetry(bytes(20))
        with self.assertRaises(FrozenInstanceError):
            telemetry.battery_voltage = 12.0
        self.assertIsInstance(telemetry.acceleration, tuple)

    def test_wrong_payload_lengths_rejected(self):
        for length in (0, 19, 21, 25):
            with self.subTest(length=length), self.assertRaises(ValueError):
                decode_telemetry(bytes(length))


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = PacketParser()
        self.expected = [(0x10, TELEMETRY_FRAME[4:-1])]

    def test_golden_telemetry_checksum_and_decode(self):
        self.assertEqual(self.parser.feed(TELEMETRY_FRAME), self.expected)

    def test_every_possible_split_preserves_frame(self):
        for split in range(len(TELEMETRY_FRAME) + 1):
            with self.subTest(split=split):
                parser = PacketParser()
                packets = parser.feed(TELEMETRY_FRAME[:split])
                packets += parser.feed(TELEMETRY_FRAME[split:])
                self.assertEqual(packets, self.expected)

    def test_bytewise_frames_and_concatenated_frames(self):
        packets = []
        for byte in TELEMETRY_FRAME:
            packets.extend(self.parser.feed(bytes((byte,))))
        self.assertEqual(packets, self.expected)
        self.assertEqual(self.parser.feed(TELEMETRY_FRAME * 3), self.expected * 3)

    def test_noise_and_partial_header_recovery(self):
        self.assertEqual(self.parser.feed(b"boot message\xaa"), [])
        self.assertEqual(self.parser.feed(b"\xaa" + TELEMETRY_FRAME), self.expected)
        self.assertEqual(self.parser.buffered_bytes, 0)

    def test_corrupt_checksum_then_valid_frame(self):
        corrupted = TELEMETRY_FRAME[:-1] + b"\xd7"
        self.assertEqual(self.parser.feed(corrupted + TELEMETRY_FRAME), self.expected)

    def test_invalid_lengths_do_not_hide_valid_frame(self):
        for length in (0, 1, 4, 38, 255):
            with self.subTest(length=length):
                parser = PacketParser()
                noise = bytes((0xAA, 0x55, length, 0x10))
                self.assertEqual(parser.feed(noise + TELEMETRY_FRAME), self.expected)

    def test_wrong_known_id_length_is_rejected_even_with_valid_checksum(self):
        bad = bytes.fromhex("AA 55 06 10 00 15")
        self.assertEqual(self.parser.feed(bad + TELEMETRY_FRAME), self.expected)

    def test_plausible_corrupt_length_recovers_without_waiting(self):
        # 37 is permitted by X-Protocol, but the incomplete garbage packet
        # cannot prevent a later complete and valid telemetry packet.
        noise = bytes.fromhex("AA 55 25 99")
        self.assertEqual(self.parser.feed(noise), [])
        self.assertEqual(self.parser.feed(TELEMETRY_FRAME), self.expected)
        self.assertEqual(self.parser.buffered_bytes, 0)

    def test_header_bytes_inside_payload_are_preserved(self):
        payload = bytes.fromhex("AA 55") + bytes(18)
        frame = bytes.fromhex("AA 55 19 10") + payload
        frame += bytes((sum(frame) & 0xFF,))
        packets = self.parser.feed(frame[:12]) + self.parser.feed(frame[12:])
        self.assertEqual(packets, [(0x10, payload)])

    def test_large_noisy_reads_leave_bounded_pending_data(self):
        self.assertEqual(self.parser.feed(b"\xaa\x55\x25\x99" * 2500), [])
        self.assertLess(self.parser.buffered_bytes, MAX_FRAME_LENGTH)
        self.assertEqual(self.parser.feed(TELEMETRY_FRAME), self.expected)

    def test_reset_discards_incomplete_frame(self):
        self.parser.feed(TELEMETRY_FRAME[:12])
        self.parser.reset()
        self.assertEqual(self.parser.buffered_bytes, 0)
        self.assertEqual(self.parser.feed(TELEMETRY_FRAME[12:]), [])
        self.assertEqual(self.parser.feed(TELEMETRY_FRAME), self.expected)

    def test_zero_payload_unknown_id_is_supported(self):
        self.assertEqual(self.parser.feed(bytes.fromhex("AA 55 05 99 9D")), [(0x99, b"")])


if __name__ == "__main__":
    unittest.main()
