import cv2
import mediapipe as mp
import math
from typing import Tuple, List, Dict, Optional, Any

class FingerCounterModel:
    """
    Model layer responsible for detecting hand landmarks using MediaPipe
    and calculating the number of extended fingers (0 to 5).
    """
    # Landmark tip IDs for [Thumb, Index, Middle, Ring, Pinky]
    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]

    def __init__(self, max_hands: int = 1, detection_con: float = 0.7, track_con: float = 0.7):
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, img_rgb: Any) -> Any:
        """Process RGB frame using MediaPipe Hands."""
        return self.hands.process(img_rgb)

    def analyze_hand(self, img: Any, results: Any) -> Optional[Dict[str, Any]]:
        """
        Analyze hand landmarks, determine open/closed status for each finger,
        and return finger states, total count, and landmark positions.
        """
        if not results.multi_hand_landmarks:
            return None

        h, w, c = img.shape
        hand_landmarks = results.multi_hand_landmarks[0]

        # Determine hand orientation (Right or Left hand)
        hand_label = "Right"
        if results.multi_handedness:
            hand_label = results.multi_handedness[0].classification[0].label

        # Draw hand skeleton connections
        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Convert normalized landmark coordinates to pixel (x, y) coordinates
        lm_list: List[Tuple[int, int]] = []
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))

        finger_states = [0, 0, 0, 0, 0]  # [Thumb, Index, Middle, Ring, Pinky]

        if lm_list:
            # 1. Thumb detection (Horizontal coordinate check based on hand type)
            # In flipped selfie view: Right hand thumb extends left (x[4] < x[3]), Left hand thumb extends right (x[4] > x[3])
            if hand_label == "Right":
                if lm_list[self.TIP_IDS[0]][0] < lm_list[self.PIP_IDS[0]][0]:
                    finger_states[0] = 1
            else:
                if lm_list[self.TIP_IDS[0]][0] > lm_list[self.PIP_IDS[0]][0]:
                    finger_states[0] = 1

            # 2. Four main fingers detection (Vertical y-coordinate check)
            for i in range(1, 5):
                tip_y = lm_list[self.TIP_IDS[i]][1]
                pip_y = lm_list[self.PIP_IDS[i]][1]
                if tip_y < pip_y:  # Finger tip is higher than PIP joint in image space
                    finger_states[i] = 1

        total_count = sum(finger_states)

        return {
            "lm_list": lm_list,
            "finger_states": finger_states,
            "total_count": total_count,
            "hand_label": hand_label,
            "tip_ids": self.TIP_IDS
        }
