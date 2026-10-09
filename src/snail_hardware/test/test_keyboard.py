"""Exercise motion lifetimes without ROS, a display, or connected hardware."""

import math
import unittest

from snail_hardware.keyboard_policy import HeldKeyDrive, PulseKeyDrive, TerminalKeys


class KeyboardPolicyTest(unittest.TestCase):
    def test_gui_release_and_focus_clear_stop_motion(self):
        drive = HeldKeyDrive()
        self.assertEqual(drive.velocity(), (0.0, 0.0))
        drive.press('w')
        drive.press('a')
        self.assertEqual(drive.velocity(), (0.10, 0.30))
        drive.release('w')
        self.assertEqual(drive.velocity(), (0.0, 0.30))
        drive.clear()
        self.assertEqual(drive.velocity(), (0.0, 0.0))

    def test_gui_stop_key_inhibits_repeats_until_released(self):
        drive = HeldKeyDrive()
        drive.press('i')
        drive.press('space')
        drive.press('i')
        self.assertEqual(drive.velocity(), (0.0, 0.0))
        drive.release('space')
        self.assertEqual(drive.velocity(), (0.0, 0.0))
        drive.press('i')
        self.assertEqual(drive.velocity(), (0.10, 0.0))

    def test_gui_opposite_keys_cancel(self):
        drive = HeldKeyDrive()
        for key in ('Up', 'Down', 'Left', 'Right'):
            drive.press(key)
        self.assertEqual(drive.velocity(), (0.0, 0.0))
        drive.release('Down')
        drive.release('Right')
        self.assertEqual(drive.velocity(), (0.10, 0.30))

    def test_twd_motion_mapping(self):
        for key, expected in (
            ('w', (0.10, 0.0)), ('s', (-0.10, 0.0)),
            ('a', (0.0, 0.30)), ('d', (0.0, -0.30)),
            ('u', (0.10, 0.30)), ('o', (0.10, -0.30)),
            ('m', (-0.10, 0.30)), ('.', (-0.10, -0.30)),
        ):
            with self.subTest(key=key):
                drive = HeldKeyDrive()
                drive.press(key)
                self.assertEqual(drive.velocity(), expected)

    def test_terminal_expires_without_another_motion_key(self):
        drive = PulseKeyDrive()
        self.assertEqual(drive.velocity(10.0), (0.0, 0.0))
        drive.press('w', 10.0)
        self.assertEqual(drive.velocity(10.30), (0.10, 0.0))
        drive.press('x', 10.31)  # Unrelated keys cannot keep motion alive.
        self.assertEqual(drive.velocity(10.36), (0.0, 0.0))
        self.assertEqual(drive.velocity(11.0), (0.0, 0.0))

    def test_terminal_repeat_refreshes_and_new_direction_replaces(self):
        drive = PulseKeyDrive()
        drive.press('i', 10.0)
        drive.press('i', 10.25)
        self.assertEqual(drive.velocity(10.50), (0.10, 0.0))
        drive.press('l', 10.51)
        self.assertEqual(drive.velocity(10.52), (0.0, -0.30))
        drive.press(' ', 10.53)
        self.assertEqual(drive.velocity(10.54), (0.0, 0.0))

    def test_invalid_or_backwards_clock_stops_active_pulse(self):
        for now in (math.nan, math.inf, -math.inf, 9.0):
            with self.subTest(now=now):
                drive = PulseKeyDrive()
                drive.press('w', 10.0)
                self.assertEqual(drive.velocity(now), (0.0, 0.0))
                self.assertEqual(drive.velocity(10.1), (0.0, 0.0))

    def test_configuration_rejects_invalid_values(self):
        for parameter in ('speed', 'turn', 'key_timeout'):
            for value in (math.nan, math.inf, -math.inf, 0.0, -0.1, None, True):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaisesRegex(ValueError, parameter):
                        PulseKeyDrive(**{parameter: value})

    def test_configuration_rejects_limits_above_caps(self):
        for parameter, value in (('speed', 0.201), ('turn', 0.601), ('key_timeout', 1.01)):
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, parameter):
                    PulseKeyDrive(**{parameter: value})

    def test_terminal_arrow_decoder_accepts_fragmented_sequences(self):
        decoder = TerminalKeys()
        self.assertEqual(decoder.feed('\x1b['), [])
        self.assertEqual(decoder.feed('Aw'), ['i', 'w'])
        self.assertEqual(decoder.feed('\x1b[B\x1b[C\x1b[D'), ['comma', 'l', 'j'])

    def test_terminal_decoder_discards_function_and_modified_arrows(self):
        decoder = TerminalKeys()
        self.assertEqual(decoder.feed('\x1b[1;5A\x1b[15~'), [])
        self.assertEqual(decoder.feed('k'), ['k'])


if __name__ == '__main__':
    unittest.main()
