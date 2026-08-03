import cv2
import mediapipe as mp
import math
from typing import Tuple, List, Dict, Optional, Any
from src import config
from src.config import AppSettings

class FingerCounterModel:
    """
    Model layer responsible for computer vision hand landmark tracking using MediaPipe.
    Counts 0-5 extended fingers dynamically adapting to Left/Right hand orientation.
    """
    # Landmark tip IDs for [Thumb, Index, Middle, Ring, Pinky]
    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]
    GESTURE_LABELS = {
        (0, 0, 0, 0, 0): "Fist",
        (0, 1, 0, 0, 0): "Point",
        (0, 1, 1, 0, 0): "Peace",
        (1, 1, 0, 0, 0): "Gun",
        (1, 1, 1, 0, 0): "Three",
        (0, 1, 1, 1, 1): "Four",
        (1, 1, 1, 1, 1): "Open Palm",
        (1, 0, 0, 0, 1): "Rock",
    }

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or AppSettings()
        self.max_hands = self.settings.max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.settings.min_detection_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence
        )

    def process_frame(self, img_rgb: Any) -> Any:
        """Process RGB frame using adaptive brightness enhancement."""
        enhanced_rgb = self._enhance_frame(img_rgb)
        return self.hands.process(enhanced_rgb)

    @staticmethod
    def _enhance_frame(img_rgb: Any) -> Any:
        """Apply light-weight adaptive enhancement only when the scene needs it."""
        brightness = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).mean()

        if brightness < config.LOW_LIGHT_THRESHOLD:
            return cv2.convertScaleAbs(
                img_rgb,
                alpha=config.ENHANCE_ALPHA,
                beta=config.ENHANCE_BETA
            )

        if brightness > config.HIGH_LIGHT_THRESHOLD:
            return cv2.convertScaleAbs(img_rgb, alpha=1.0, beta=config.BRIGHT_SCENE_BETA)

        return img_rgb

    @staticmethod
    def _joint_angle(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float]) -> float:
        """Return the angle ABC in degrees using 3D landmarks."""
        ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
        bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

        mag_ba = math.sqrt(sum(component * component for component in ba))
        mag_bc = math.sqrt(sum(component * component for component in bc))
        if mag_ba == 0 or mag_bc == 0:
            return 0.0

        cosine = sum(ba[i] * bc[i] for i in range(3)) / (mag_ba * mag_bc)
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    @classmethod
    def _estimate_stability(cls, landmarks: List[Tuple[float, float, float]]) -> float:
        """Estimate pose stability from palm size and finger spread."""
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]

        palm_size = cls._distance(wrist, middle_mcp)
        finger_spread = cls._distance(index_tip, pinky_tip)
        if palm_size == 0:
            return 0.0

        score = finger_spread / palm_size
        return max(0.0, min(1.0, score / 2.5))

    @classmethod
    def _describe_gesture(cls, finger_states: List[int]) -> str:
        """Map common finger combinations to a friendly gesture label."""
        return cls.GESTURE_LABELS.get(tuple(finger_states), f"Count {sum(finger_states)}")

    @classmethod
    def _is_thumb_extended(cls, landmarks: List[Tuple[float, float, float]]) -> bool:
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]

        thumb_angle = cls._joint_angle(thumb_mcp, thumb_ip, thumb_tip)
        thumb_reach = cls._distance(thumb_tip, wrist) - cls._distance(thumb_ip, wrist)
        palm_spread = cls._distance(thumb_tip, landmarks[5]) - cls._distance(thumb_ip, landmarks[5])

        return (
            thumb_angle >= config.THUMB_ANGLE_THRESHOLD and
            thumb_reach > config.DISTANCE_MARGIN and
            palm_spread > 0
        )

    @classmethod
    def _is_finger_extended(cls, landmarks: List[Tuple[float, float, float]], tip_id: int, pip_id: int) -> bool:
        wrist = landmarks[0]
        mcp_id = pip_id - 1
        dip_id = tip_id - 1

        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        mcp = landmarks[mcp_id]
        dip = landmarks[dip_id]

        pip_angle = cls._joint_angle(mcp, pip, dip)
        tip_reach = cls._distance(tip, wrist) - cls._distance(pip, wrist)

        return pip_angle >= config.FINGER_ANGLE_THRESHOLD and tip_reach > config.DISTANCE_MARGIN

    @staticmethod
    def _select_primary_hand(results: Any) -> Tuple[Any, str]:
        """Choose the most prominent hand when multiple detections are available."""
        if not results.multi_hand_landmarks:
            return None, "Hand"

        handedness_labels = []
        if results.multi_handedness:
            for handedness in results.multi_handedness:
                label = handedness.classification[0].label
                handedness_labels.append("Right" if label == "Left" else "Left")
        else:
            handedness_labels = ["Hand"] * len(results.multi_hand_landmarks)

        best_score = None
        selected_hand = results.multi_hand_landmarks[0]
        selected_label = handedness_labels[0] if handedness_labels else "Hand"

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            center_distance = math.hypot(center_x - 0.5, center_y - 0.5)
            score = bbox_area - (center_distance * 0.05)

            if best_score is None or score > best_score:
                best_score = score
                selected_hand = hand_landmarks
                selected_label = handedness_labels[idx] if idx < len(handedness_labels) else "Hand"

        return selected_hand, selected_label

    def analyze_hand(self, img: Any, results: Any) -> Optional[Dict[str, Any]]:
        """
        Analyze hand landmarks, determine open/closed status for each finger,
        and return finger states, total count, and landmark positions.
        """
        if not results.multi_hand_landmarks:
            return None

        h, w, c = img.shape
        hand_landmarks, hand_label = self._select_primary_hand(results)

        # Convert normalized landmark coordinates to pixel (x, y) coordinates
        lm_list: List[Tuple[int, int]] = []
        normalized_lm_list: List[Tuple[float, float, float]] = []
        out_of_bounds = False
        w_margin = w * 0.05
        h_margin = h * 0.05
        
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))
            normalized_lm_list.append((lm.x, lm.y, lm.z))
            if cx < w_margin or cx > w - w_margin or cy < h_margin or cy > h - h_margin:
                out_of_bounds = True

        finger_states = [0, 0, 0, 0, 0]  # [Thumb, Index, Middle, Ring, Pinky]

        if len(lm_list) >= 21:
            if self._is_thumb_extended(normalized_lm_list):
                finger_states[0] = 1

            for i in range(1, 5):
                if self._is_finger_extended(normalized_lm_list, self.TIP_IDS[i], self.PIP_IDS[i]):
                    finger_states[i] = 1

        total_count = sum(finger_states)
        stability = self._estimate_stability(normalized_lm_list)
        gesture_label = self._describe_gesture(finger_states)
        xs = [point[0] for point in lm_list]
        ys = [point[1] for point in lm_list]
        bbox = {
            "x": min(xs),
            "y": min(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
        }

        return {
            "hand_landmarks": hand_landmarks,
            "lm_list": lm_list,
            "finger_states": finger_states,
            "total_count": total_count,
            "hand_label": hand_label,
            "gesture_label": gesture_label,
            "stability": stability,
            "bounding_box": bbox,
            "tip_ids": self.TIP_IDS,
            "out_of_bounds": out_of_bounds
        }

    def close(self) -> None:
        """Release MediaPipe graph resources explicitly."""
        self.hands.close()
