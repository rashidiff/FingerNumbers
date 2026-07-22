from typing import Tuple

"""
Configuration settings and constants for the Finger Counter Application.
"""

# Camera Settings
CAMERA_INDEX: int = 0
WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720
WINDOW_NAME: str = "Finger Counter"

# MediaPipe Settings
MIN_DETECTION_CONFIDENCE: float = 0.3
MIN_TRACKING_CONFIDENCE: float = 0.3
MAX_HANDS: int = 1

# Image Enhancement
ENHANCE_ALPHA: float = 1.2
ENHANCE_BETA: int = 30

# Colors (BGR format for OpenCV)
COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)
COLOR_RED: Tuple[int, int, int] = (0, 0, 255)
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_BLACK: Tuple[int, int, int] = (0, 0, 0)
COLOR_ORANGE: Tuple[int, int, int] = (0, 165, 255)
COLOR_BLUE: Tuple[int, int, int] = (255, 0, 0)
COLOR_GRAY: Tuple[int, int, int] = (100, 100, 100)

# HUD Settings
HUD_BOX_START: Tuple[int, int] = (40, 40)
HUD_BOX_END: Tuple[int, int] = (220, 200)
COUNT_TEXT_POS: Tuple[int, int] = (105, 155)
LABEL_TEXT_POS: Tuple[int, int] = (45, 230)
FPS_TEXT_POS: Tuple[int, int] = (1050, 50)
