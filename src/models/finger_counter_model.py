"""MediaPipe-backed hand landmark analysis."""

import math
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp

from src import config
from src.config import AppSettings
from src.models.types import BoundingBox, HandObservation


class FingerCounterModel:
    """Detect and classify every hand present in a frame."""

    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]
    GESTURE_LABELS = {
        (0, 0, 0, 0, 0): "Fist", (0, 1, 0, 0, 0): "Point",
        (0, 1, 1, 0, 0): "Peace", (1, 1, 0, 0, 0): "Gun",
        (1, 1, 1, 0, 0): "Three", (0, 1, 1, 1, 1): "Four",
        (1, 1, 1, 1, 1): "Open Palm", (1, 0, 0, 0, 1): "Rock",
    }

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or AppSettings()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=self.settings.max_hands,
            min_detection_confidence=self.settings.min_detection_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence,
        )

    def process_frame(self, img_rgb: Any) -> Any:
        return self.hands.process(self._enhance_frame(img_rgb))

    @staticmethod
    def _enhance_frame(img_rgb: Any) -> Any:
        brightness = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).mean()
        if brightness < config.LOW_LIGHT_THRESHOLD:
            return cv2.convertScaleAbs(img_rgb, alpha=config.ENHANCE_ALPHA, beta=config.ENHANCE_BETA)
        if brightness > config.HIGH_LIGHT_THRESHOLD:
            return cv2.convertScaleAbs(img_rgb, alpha=1.0, beta=config.BRIGHT_SCENE_BETA)
        return img_rgb

    @staticmethod
    def _joint_angle(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float]) -> float:
        ba = tuple(a[i] - b[i] for i in range(3))
        bc = tuple(c[i] - b[i] for i in range(3))
        mag_ba = math.sqrt(sum(value * value for value in ba))
        mag_bc = math.sqrt(sum(value * value for value in bc))
        if mag_ba == 0 or mag_bc == 0:
            return 0.0
        cosine = sum(ba[i] * bc[i] for i in range(3)) / (mag_ba * mag_bc)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    @classmethod
    def _estimate_stability(cls, landmarks: List[Tuple[float, float, float]]) -> float:
        palm_size = cls._distance(landmarks[0], landmarks[9])
        if palm_size == 0:
            return 0.0
        spread = cls._distance(landmarks[8], landmarks[20])
        return max(0.0, min(1.0, (spread / palm_size) / 2.5))

    @classmethod
    def _describe_gesture(cls, finger_states: List[int]) -> str:
        return cls.GESTURE_LABELS.get(tuple(finger_states), f"Count {sum(finger_states)}")

    @classmethod
    def _is_thumb_extended(cls, landmarks: List[Tuple[float, float, float]]) -> bool:
        angle = cls._joint_angle(landmarks[2], landmarks[3], landmarks[4])
        reach = cls._distance(landmarks[4], landmarks[0]) - cls._distance(landmarks[3], landmarks[0])
        spread = cls._distance(landmarks[4], landmarks[5]) - cls._distance(landmarks[3], landmarks[5])
        return angle >= config.THUMB_ANGLE_THRESHOLD and reach > config.DISTANCE_MARGIN and spread > 0

    @classmethod
    def _is_finger_extended(cls, landmarks: List[Tuple[float, float, float]], tip_id: int, pip_id: int) -> bool:
        angle = cls._joint_angle(landmarks[pip_id - 1], landmarks[pip_id], landmarks[tip_id - 1])
        reach = cls._distance(landmarks[tip_id], landmarks[0]) - cls._distance(landmarks[pip_id], landmarks[0])
        return angle >= config.FINGER_ANGLE_THRESHOLD and reach > config.DISTANCE_MARGIN

    @staticmethod
    def _hand_labels(results: Any) -> List[str]:
        if not results.multi_handedness:
            return ["Hand"] * len(results.multi_hand_landmarks)
        return ["Right" if item.classification[0].label == "Left" else "Left" for item in results.multi_handedness]

    @classmethod
    def _select_primary_index(cls, results: Any) -> int:
        if not results.multi_hand_landmarks:
            return -1
        best_index, best_score = 0, None
        for index, hand in enumerate(results.multi_hand_landmarks):
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            center_distance = math.hypot(sum(xs) / len(xs) - 0.5, sum(ys) / len(ys) - 0.5)
            score = area - center_distance * 0.05
            if best_score is None or score > best_score:
                best_index, best_score = index, score
        return best_index

    @classmethod
    def _select_primary_hand(cls, results: Any) -> Tuple[Any, str]:
        index = cls._select_primary_index(results)
        if index < 0:
            return None, "Hand"
        return results.multi_hand_landmarks[index], cls._hand_labels(results)[index]

    def analyze_hands(self, img: Any, results: Any) -> List[HandObservation]:
        """Return an observation for every detected hand, preserving order."""
        if not results.multi_hand_landmarks:
            return []
        height, width = img.shape[:2]
        labels = self._hand_labels(results)
        observations: List[HandObservation] = []
        for index, hand in enumerate(results.multi_hand_landmarks):
            pixels, normalized, out_of_bounds = [], [], False
            for landmark in hand.landmark:
                point = (int(landmark.x * width), int(landmark.y * height))
                pixels.append(point)
                normalized.append((landmark.x, landmark.y, landmark.z))
                out_of_bounds |= point[0] < width * 0.05 or point[0] > width * 0.95
                out_of_bounds |= point[1] < height * 0.05 or point[1] > height * 0.95
            if len(pixels) < 21:
                continue
            states = [int(self._is_thumb_extended(normalized))]
            states.extend(int(self._is_finger_extended(normalized, self.TIP_IDS[i], self.PIP_IDS[i])) for i in range(1, 5))
            xs, ys = zip(*pixels)
            observations.append(HandObservation(
                hand_landmarks=hand, landmarks=pixels, normalized_landmarks=normalized,
                finger_states=states, total_count=sum(states),
                hand_label=labels[index] if index < len(labels) else "Hand",
                gesture_label=self._describe_gesture(states),
                stability=self._estimate_stability(normalized),
                bounding_box=BoundingBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                out_of_bounds=out_of_bounds,
            ))
        return observations

    def analyze_hand(self, img: Any, results: Any) -> Optional[HandObservation]:
        """Compatibility API returning the primary hand only."""
        observations = self.analyze_hands(img, results)
        if not observations:
            return None
        return observations[min(self._select_primary_index(results), len(observations) - 1)]

    def close(self) -> None:
        self.hands.close()
