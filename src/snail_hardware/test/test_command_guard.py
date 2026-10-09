import unittest

from snail_hardware.command_guard import CommandGuard


class CommandGuardTests(unittest.TestCase):
    def test_commands_wait_for_feedback(self):
        guard = CommandGuard()
        self.assertFalse(guard.accept(0.1, 0.3, 1.0))
        guard.feedback(1.1)
        self.assertEqual(guard.output(1.1), (0.0, 0.0))
        self.assertTrue(guard.accept(0.1, 0.3, 1.2))
        self.assertEqual(guard.output(1.3), (0.1, 0.3))

    def test_command_timeout_even_with_live_feedback(self):
        guard = CommandGuard()
        guard.feedback(1.0)
        guard.accept(0.1, 0.3, 1.0)
        guard.feedback(1.31)
        self.assertEqual(guard.output(1.31), (0.0, 0.0))

    def test_feedback_loss_and_recovery_never_replays_old_motion(self):
        guard = CommandGuard(command_timeout=2.0)
        guard.feedback(1.0)
        guard.accept(0.1, 0.3, 1.0)
        self.assertEqual(guard.output(1.51), (0.0, 0.0))
        guard.feedback(1.52)
        self.assertEqual(guard.output(1.52), (0.0, 0.0))
        guard.accept(0.1, 0.3, 1.53)
        self.assertEqual(guard.output(1.54), (0.1, 0.3))

    def test_recovery_clears_old_motion_without_a_timeout_tick(self):
        guard = CommandGuard(command_timeout=2.0)
        guard.feedback(1.0)
        guard.accept(0.1, 0.3, 1.0)
        guard.feedback(2.0)
        self.assertEqual(guard.output(2.0), (0.0, 0.0))

    def test_clamp_and_invalid_commands(self):
        guard = CommandGuard()
        guard.feedback(1.0)
        guard.accept(4.0, -4.0, 1.0)
        self.assertEqual(guard.output(1.0), (0.2, -0.6))
        self.assertFalse(guard.accept(float('nan'), 0.0, 1.0))
        self.assertEqual(guard.output(1.0), (0.0, 0.0))

    def test_invalid_configuration(self):
        for options in ({'command_timeout': 0}, {'feedback_timeout': float('nan')},
                        {'max_linear': 2.0}, {'max_angular': 7.0}):
            with self.assertRaises(ValueError):
                CommandGuard(**options)
