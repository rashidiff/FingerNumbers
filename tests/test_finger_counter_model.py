import unittest
from collections import deque
from types import SimpleNamespace

from main import parse_settings
from src import config
from src.controllers.main_controller import MainController
from src.models.finger_counter_model import FingerCounterModel


def build_landmarks(overrides=None):
    landmarks = [(0.0, 0.0, 0.0)] * 21
    overrides = overrides or {}
    for index, value in overrides.items():
        landmarks[index] = value
    return landmarks


def build_result(hands, labels):
    handedness = [
        SimpleNamespace(classification=[SimpleNamespace(label=label)])
        for label in labels
    ]
    return SimpleNamespace(
        multi_hand_landmarks=hands,
        multi_handedness=handedness
    )


class FingerCounterModelTests(unittest.TestCase):
    def test_analyze_hands_preserves_all_detected_hands(self):
        first = SimpleNamespace(landmark=[SimpleNamespace(x=0.25, y=0.4, z=0.0) for _ in range(21)])
        second = SimpleNamespace(landmark=[SimpleNamespace(x=0.75, y=0.4, z=0.0) for _ in range(21)])
        result = build_result([first, second], ["Left", "Right"])
        model = FingerCounterModel.__new__(FingerCounterModel)

        observations = model.analyze_hands(SimpleNamespace(shape=(480, 640, 3)), result)

        self.assertEqual(len(observations), 2)
        self.assertEqual([item.hand_label for item in observations], ["Right", "Left"])
        self.assertEqual([item.total_count for item in observations], [0, 0])

    def test_joint_angle_for_straight_line_is_180(self):
        angle = FingerCounterModel._joint_angle(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0)
        )
        self.assertAlmostEqual(angle, 180.0)

    def test_index_finger_extension_detection(self):
        landmarks = build_landmarks({
            0: (0.0, 1.0, 0.0),
            5: (0.0, 0.7, 0.0),
            6: (0.0, 0.5, 0.0),
            7: (0.0, 0.25, 0.0),
            8: (0.0, 0.0, 0.0),
        })
        self.assertTrue(FingerCounterModel._is_finger_extended(landmarks, 8, 6))

    def test_index_finger_bend_is_not_counted_as_open(self):
        landmarks = build_landmarks({
            0: (0.0, 1.0, 0.0),
            5: (0.0, 0.7, 0.0),
            6: (0.0, 0.5, 0.0),
            7: (0.2, 0.45, 0.0),
            8: (0.25, 0.55, 0.0),
        })
        self.assertFalse(FingerCounterModel._is_finger_extended(landmarks, 8, 6))

    def test_thumb_extension_detection(self):
        landmarks = build_landmarks({
            0: (0.1, 0.7, 0.0),
            1: (0.2, 0.65, 0.0),
            2: (0.3, 0.6, 0.0),
            3: (0.45, 0.58, 0.0),
            4: (0.65, 0.57, 0.0),
            5: (0.35, 0.55, 0.0),
        })
        self.assertTrue(FingerCounterModel._is_thumb_extended(landmarks))

    def test_primary_hand_prefers_larger_centered_detection(self):
        large_center_hand = SimpleNamespace(
            landmark=[SimpleNamespace(x=0.35, y=0.3, z=0.0) for _ in range(21)]
        )
        for idx, x_value in enumerate([0.25, 0.75, 0.28, 0.72]):
            large_center_hand.landmark[idx].x = x_value
        for idx, y_value in enumerate([0.2, 0.8, 0.22, 0.78]):
            large_center_hand.landmark[idx + 4].y = y_value

        small_edge_hand = SimpleNamespace(
            landmark=[SimpleNamespace(x=0.85, y=0.1, z=0.0) for _ in range(21)]
        )
        result = build_result([small_edge_hand, large_center_hand], ["Right", "Left"])

        selected_hand, hand_label = FingerCounterModel._select_primary_hand(result)

        self.assertIs(selected_hand, large_center_hand)
        self.assertEqual(hand_label, "Right")


class MainControllerSmoothingTests(unittest.TestCase):
    def test_majority_vote_stabilizes_finger_states(self):
        controller = MainController.__new__(MainController)
        controller.finger_history = deque(maxlen=config.SMOOTHING_WINDOW)

        stable_hand = {
            "finger_states": [1, 1, 0, 0, 0],
            "total_count": 2,
            "hand_label": "Right",
            "lm_list": [],
            "tip_ids": [],
            "out_of_bounds": False,
            "hand_landmarks": None,
        }
        noisy_hand = dict(stable_hand)
        noisy_hand["finger_states"] = [1, 0, 0, 0, 0]
        noisy_hand["total_count"] = 1

        controller._stabilize_hand_data(stable_hand)
        controller._stabilize_hand_data(stable_hand)
        stabilized = controller._stabilize_hand_data(noisy_hand)

        self.assertEqual(stabilized["finger_states"], [1, 1, 0, 0, 0])
        self.assertEqual(stabilized["total_count"], 2)


class MainCliSettingsTests(unittest.TestCase):
    def test_parse_settings_overrides_defaults(self):
        settings = parse_settings([
            "--camera", "2",
            "--width", "960",
            "--height", "540",
            "--max-hands", "2",
            "--detection-confidence", "0.6",
            "--tracking-confidence", "0.7",
            "--smoothing-window", "9",
            "--hide-diagnostics",
            "--hide-skeleton",
        ])

        self.assertEqual(settings.camera_index, 2)
        self.assertEqual(settings.window_width, 960)
        self.assertEqual(settings.window_height, 540)
        self.assertEqual(settings.max_hands, 2)
        self.assertEqual(settings.min_detection_confidence, 0.6)
        self.assertEqual(settings.min_tracking_confidence, 0.7)
        self.assertEqual(settings.smoothing_window, 9)
        self.assertFalse(settings.show_diagnostics)
        self.assertTrue(settings.show_controls)
        self.assertFalse(settings.show_skeleton)


if __name__ == "__main__":
    unittest.main()
