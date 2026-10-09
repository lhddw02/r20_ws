"""Exercise motion safety transitions without ROS or actual wheel commands."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snail_bottle_approach.approach_policy import ApproachPolicy, Target


def target(distance=1.5, bearing=0.0, bbox=(10.0, 10.0, 50.0, 90.0),
           confidence=0.9):
    return Target(bbox, distance, bearing, confidence)


class PolicyTests(unittest.TestCase):
    def confirmed(self, item=None, **configuration):
        policy = ApproachPolicy(**configuration)
        item = item or target()
        for now in (1.0, 1.1, 1.2):
            policy.update(item, now)
        return policy

    def test_initially_stopped(self):
        policy = ApproachPolicy()
        self.assertEqual(policy.state, 'WAITING')
        self.assertEqual(policy.command(0.0), (0.0, 0.0))

    def test_three_distinct_frames_are_required(self):
        policy = ApproachPolicy()
        self.assertEqual(policy.confirmation_count, 0)
        policy.update(target(), 1.0)
        policy.update(target(), 1.0)
        policy.update(target(), 1.0)
        self.assertEqual(policy.confirmation_count, 1)
        self.assertEqual(policy.command(1.0), (0.0, 0.0))
        self.assertEqual(policy.state, 'CONFIRMING')
        policy.update(target(), 1.1)
        self.assertEqual(policy.confirmation_count, 2)
        self.assertEqual(policy.command(1.1), (0.0, 0.0))
        policy.update(target(), 1.2)
        self.assertEqual(policy.confirmation_count, 3)
        self.assertEqual(policy.state, 'APPROACHING')
        self.assertEqual(policy.command(1.2), (0.08, 0.0))
        with self.assertRaises(AttributeError):
            policy.confirmation_count = 10

    def test_small_bbox_jitter_remains_associated(self):
        policy = ApproachPolicy()
        for index, shift in enumerate((0, 3, -3)):
            policy.update(target(bbox=(10 + shift, 10, 50 + shift, 90)),
                          1.0 + index * 0.1)
        self.assertEqual(policy.state, 'APPROACHING')

    def test_new_target_requires_confirmation(self):
        policy = self.confirmed()
        alternate = target(bbox=(200, 10, 240, 90))
        policy.update(alternate, 1.3)
        self.assertEqual(policy.command(1.3), (0.0, 0.0))
        self.assertEqual(policy.state, 'CONFIRMING')
        policy.update(alternate, 1.4)
        self.assertEqual(policy.command(1.4), (0.0, 0.0))
        policy.update(alternate, 1.5)
        self.assertGreater(policy.command(1.5)[0], 0)

    def test_detection_loss_stops_immediately_and_reconfirms(self):
        policy = self.confirmed()
        policy.update(None, 1.21)
        self.assertEqual(policy.command(1.21), (0.0, 0.0))
        policy.update(target(), 1.3)
        policy.update(target(), 1.4)
        self.assertEqual(policy.command(1.4), (0.0, 0.0))

    def test_stale_target_stops_and_clears_confirmations(self):
        policy = self.confirmed()
        self.assertGreater(policy.command(1.54)[0], 0)
        self.assertEqual(policy.command(1.56), (0.0, 0.0))
        self.assertEqual(policy.state, 'WAITING')
        policy.update(target(), 1.6)
        self.assertEqual(policy.command(1.6), (0.0, 0.0))

    def test_gap_between_frames_restarts_confirmation(self):
        policy = self.confirmed()
        policy.update(target(), 1.7)
        self.assertEqual(policy.state, 'CONFIRMING')
        self.assertEqual(policy.command(1.7), (0.0, 0.0))

    def test_slow_fresh_results_confirm_without_moving_during_expiry_gaps(self):
        # CPU inference can finish within the deadline yet leave zero-command
        # intervals between results. These intervals must not erase evidence
        # from distinct observations whose original timestamps remain close.
        for latency in (0.28, 0.35):
            with self.subTest(latency=latency):
                policy = ApproachPolicy(target_timeout=0.50)
                result_index = 0
                last_capture = None
                moving_ticks = stale_ticks = 0
                for tick in range(1, 61):
                    now = 10.0 + tick * 0.05
                    completion = 10.0 + (result_index + 1) * latency
                    if now + 1e-9 >= completion:
                        last_capture = 10.0 + result_index * latency
                        policy.update(target(distance=1.2), last_capture)
                        result_index += 1
                    command = policy.command(now, preserve_confirmations=True)
                    if last_capture is None or now - last_capture >= 0.50:
                        stale_ticks += 1
                        self.assertEqual(command, (0.0, 0.0))
                    elif result_index < 3:
                        self.assertEqual(command, (0.0, 0.0))
                    else:
                        moving_ticks += 1
                        self.assertGreater(command[0], 0.0)
                        self.assertEqual(command[1], 0.0)
                self.assertGreater(moving_ticks, 0)
                self.assertGreater(stale_ticks, 0)

    def test_expired_inference_never_commands_motion_even_with_retained_history(self):
        for latency in (0.50, 0.55):
            with self.subTest(latency=latency):
                policy = ApproachPolicy(target_timeout=0.50)
                for index in range(6):
                    capture = 10.0 + index * latency
                    policy.update(target(distance=1.2), capture)
                    command = policy.command(capture + latency,
                                             preserve_confirmations=True)
                    self.assertEqual(command, (0.0, 0.0))

    def test_retained_history_stops_at_exact_deadline_without_clearing_confirmation(self):
        policy = self.confirmed(target_timeout=0.50)
        self.assertEqual(policy.command(1.70, preserve_confirmations=True),
                         (0.0, 0.0))
        self.assertEqual(policy.state, 'APPROACHING')
        self.assertEqual(policy.confirmation_count, 3)
        policy.update(target(), 1.45)
        self.assertEqual(policy.confirmation_count, 4)
        self.assertGreater(policy.command(1.73, preserve_confirmations=True)[0], 0)

    def test_real_observation_outage_requires_three_new_confirmations(self):
        policy = self.confirmed(target_timeout=0.50)
        self.assertEqual(policy.command(2.0, preserve_confirmations=True),
                         (0.0, 0.0))
        for capture in (2.1, 2.38):
            policy.update(target(), capture)
            self.assertEqual(policy.state, 'CONFIRMING')
            self.assertEqual(policy.command(capture + 0.28, preserve_confirmations=True),
                             (0.0, 0.0))
        policy.update(target(), 2.66)
        self.assertGreater(policy.command(2.94, preserve_confirmations=True)[0], 0)

    def test_missing_or_invalid_target_clears_preserved_history(self):
        for item in (None, target(distance=math.nan), target(confidence=0)):
            with self.subTest(item=item):
                policy = self.confirmed(target_timeout=0.50)
                policy.command(1.75, preserve_confirmations=True)
                policy.update(item, 1.8)
                self.assertEqual(policy.state, 'WAITING')
                for capture in (1.9, 2.0):
                    policy.update(target(), capture)
                    self.assertEqual(policy.command(capture, preserve_confirmations=True),
                                     (0.0, 0.0))
                policy.update(target(), 2.1)
                self.assertGreater(policy.command(2.1, preserve_confirmations=True)[0], 0)

    def test_retaining_history_does_not_allow_invalid_or_regressing_clock(self):
        for now in (math.nan, math.inf, 1.1):
            with self.subTest(now=now):
                policy = self.confirmed()
                self.assertEqual(policy.command(now, preserve_confirmations=True),
                                 (0.0, 0.0))
                self.assertEqual(policy.state, 'WAITING')

    def test_right_bearing_turns_right_without_forward_motion(self):
        policy = self.confirmed(target(bearing=0.5))
        self.assertEqual(policy.command(1.2), (0.0, -0.30))

    def test_left_bearing_turns_left_without_forward_motion(self):
        policy = self.confirmed(target(bearing=-0.5))
        self.assertEqual(policy.command(1.2), (0.0, 0.30))

    def test_alignment_deadband_allows_forward_motion(self):
        for bearing in (-0.15, 0, 0.15):
            with self.subTest(bearing=bearing):
                self.assertEqual(self.confirmed(target(bearing=bearing)).command(1.2),
                                 (0.08, 0.0))

    def test_speed_scales_near_stop_without_reversing(self):
        policy = self.confirmed(target(distance=0.65))
        self.assertAlmostEqual(policy.command(1.2)[0], 0.04)
        policy.update(target(distance=0.401), 1.3)
        self.assertEqual(policy.command(1.3), (0.01, 0.0))
        policy.update(target(distance=0.39), 1.4)
        self.assertEqual(policy.command(1.4), (0.0, 0.0))

    def test_near_stop_progress_survives_millimetre_per_second_truncation(self):
        policy = self.confirmed(target(distance=0.400001))
        speed, angular = policy.command(1.2)
        self.assertEqual(int(speed * 1000), 10)
        self.assertEqual(angular, 0.0)
        self.assertLessEqual(speed, policy.max_linear)
        policy.update(target(distance=0.40), 1.3)
        self.assertEqual(policy.state, 'REACHED')
        self.assertEqual(policy.command(1.3), (0.0, 0.0))

    def test_minimum_speed_respects_a_lower_maximum_speed(self):
        policy = self.confirmed(target(distance=0.400001), max_linear=0.005)
        self.assertEqual(policy.command(1.2), (0.005, 0.0))
        policy.update(target(distance=2.0), 1.3)
        self.assertEqual(policy.command(1.3), (0.005, 0.0))
        policy.update(target(distance=0.39), 1.4)
        self.assertEqual(policy.command(1.4), (0.0, 0.0))

    def test_stop_distance_is_inclusive_even_when_off_center(self):
        policy = self.confirmed(target(distance=0.40, bearing=0.6))
        self.assertEqual(policy.state, 'REACHED')
        self.assertEqual(policy.command(1.2), (0.0, 0.0))

    def test_reached_latches_until_explicit_reset(self):
        policy = self.confirmed(target(distance=0.3))
        policy.update(None, 1.3)
        policy.update(target(), 2.0)
        self.assertEqual(policy.command(10.0), (0.0, 0.0))
        self.assertEqual(policy.state, 'REACHED')
        policy.reset()
        policy.update(target(), 10.1)
        self.assertEqual(policy.state, 'CONFIRMING')
        self.assertEqual(policy.command(10.1), (0.0, 0.0))

    def test_invalid_targets_stop(self):
        invalid = (target(distance=math.nan), target(distance=math.inf),
                   target(distance=-0.1), target(bearing=math.inf),
                   target(bearing=math.pi), target(confidence=0),
                   target(confidence=1.1), target(bbox=(10, 10, 10, 50)),
                   target(bbox=(-1, 0, 10, 50)),
                   target(bbox=(0, 0, math.nan, 50)))
        for item in invalid:
            with self.subTest(item=item):
                policy = self.confirmed()
                policy.update(item, 1.3)
                self.assertEqual(policy.command(1.3), (0.0, 0.0))

    def test_invalid_or_regressing_clock_stops(self):
        for now in (math.nan, math.inf, 1.1):
            with self.subTest(now=now):
                policy = self.confirmed()
                self.assertEqual(policy.command(now), (0.0, 0.0))
                self.assertEqual(policy.state, 'WAITING')
                policy = self.confirmed()
                policy.update(target(), now)
                self.assertEqual(policy.command(1.3), (0.0, 0.0))

    def test_invalid_configuration_is_rejected(self):
        for config in ({'max_linear': -0.1}, {'max_angular': math.nan},
                       {'confirmations': 0}, {'confirmations': 3.5},
                       {'confirmations': True}, {'confirmation_iou': 0},
                       {'alignment_threshold': math.pi},
                       {'min_linear': 0}, {'min_linear': math.nan}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    ApproachPolicy(**config)


if __name__ == '__main__':
    unittest.main()
