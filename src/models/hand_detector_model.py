import cv2
import mediapipe as mp
import math
from typing import Tuple, Optional, Dict, Any

class HandDetectorModel:
    """
    Model layer responsible for computer vision hand landmark tracking using MediaPipe.
    """
    def __init__(self, mode: bool = False, max_hands: int = 1, detection_con: float = 0.7, track_con: float = 0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, img_rgb) -> Any:
        """Process RGB image and return raw hand detection results."""
        return self.hands.process(img_rgb)

    def extract_pinch_landmarks(self, img, results) -> Optional[Dict[str, Any]]:
        """
        Extract Thumb tip (landmark 4) and Index tip (landmark 8) coordinates,
        and calculate Euclidean distance and midpoint.
        """
        if not results.multi_hand_landmarks:
            return None

        h, w, c = img.shape
        # Process primary hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Optional: draw full hand skeleton connections
        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Landmark 4: Thumb Tip, Landmark 8: Index Finger Tip
        lm_thumb = hand_landmarks.landmark[4]
        lm_index = hand_landmarks.landmark[8]

        x1, y1 = int(lm_thumb.x * w), int(lm_thumb.y * h)
        x2, y2 = int(lm_index.x * w), int(lm_index.y * h)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Euclidean distance between Thumb and Index tips
        distance = math.hypot(x2 - x1, y2 - y1)

        return {
            "p1": (x1, y1),
            "p2": (x2, y2),
            "center": (cx, cy),
            "distance": distance
        }
