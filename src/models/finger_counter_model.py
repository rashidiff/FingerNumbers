import cv2
import mediapipe as mp
import math
from typing import Tuple, List, Dict, Optional, Any
from src import config

class FingerCounterModel:
    """
    Model layer responsible for computer vision hand landmark tracking using MediaPipe.
    Counts 0-5 extended fingers dynamically adapting to Left/Right hand orientation.
    """
    # Landmark tip IDs for [Thumb, Index, Middle, Ring, Pinky]
    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]

    def __init__(self, max_hands: int = config.MAX_HANDS, detection_con: float = config.MIN_DETECTION_CONFIDENCE, track_con: float = config.MIN_TRACKING_CONFIDENCE):
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=detection_con,  # Very low threshold for difficult lighting
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, img_rgb: Any) -> Any:
        """Process RGB frame using MediaPipe Hands with brightness enhancement."""
        # Enhance brightness and contrast to help MediaPipe detect dark/backlit hands
        enhanced_rgb = cv2.convertScaleAbs(img_rgb, alpha=config.ENHANCE_ALPHA, beta=config.ENHANCE_BETA)
        return self.hands.process(enhanced_rgb)

    def analyze_hand(self, img: Any, results: Any) -> Optional[Dict[str, Any]]:
        """
        Analyze hand landmarks, determine open/closed status for each finger,
        and return finger states, total count, and landmark positions.
        """
        if not results.multi_hand_landmarks:
            return None

        h, w, c = img.shape
        hand_landmarks = results.multi_hand_landmarks[0]

        # Determine hand orientation label
        hand_label = "Hand"
        if results.multi_handedness:
            # Note: in selfie view (flipped), MediaPipe label is mirrored
            raw_label = results.multi_handedness[0].classification[0].label
            hand_label = "Right" if raw_label == "Left" else "Left"

        # Draw hand skeleton connections
        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Convert normalized landmark coordinates to pixel (x, y) coordinates
        lm_list: List[Tuple[int, int]] = []
        out_of_bounds = False
        w_margin = w * 0.05
        h_margin = h * 0.05
        
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))
            if cx < w_margin or cx > w - w_margin or cy < h_margin or cy > h - h_margin:
                out_of_bounds = True

        finger_states = [0, 0, 0, 0, 0]  # [Thumb, Index, Middle, Ring, Pinky]

        if len(lm_list) >= 21:
            # 1. Thumb Detection: Compare distance between Thumb Tip (4) -> Pinky MCP (17) vs Thumb IP (3) -> Pinky MCP (17)
            # This Euclidean distance check is rotation-invariant and works for both Left and Right hands perfectly.
            dist_thumb_tip = math.hypot(lm_list[4][0] - lm_list[17][0], lm_list[4][1] - lm_list[17][1])
            dist_thumb_ip = math.hypot(lm_list[3][0] - lm_list[17][0], lm_list[3][1] - lm_list[17][1])
            if dist_thumb_tip > dist_thumb_ip:
                finger_states[0] = 1

            # 2. Four Main Fingers Detection (Index, Middle, Ring, Pinky)
            # Check if Finger Tip (8, 12, 16, 20) is higher (smaller y) than PIP Joint (6, 10, 14, 18)
            for i in range(1, 5):
                tip_y = lm_list[self.TIP_IDS[i]][1]
                pip_y = lm_list[self.PIP_IDS[i]][1]
                if tip_y < pip_y:
                    finger_states[i] = 1

        total_count = sum(finger_states)

        return {
            "lm_list": lm_list,
            "finger_states": finger_states,
            "total_count": total_count,
            "hand_label": hand_label,
            "tip_ids": self.TIP_IDS,
            "out_of_bounds": out_of_bounds
        }
