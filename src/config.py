"""
Configuration settings and constants for the Finger Counter Application.
"""

# Camera Settings
CAMERA_INDEX = 0
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_NAME = "Finger Counter"

# MediaPipe Settings
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3
MAX_HANDS = 1

# Image Enhancement
ENHANCE_ALPHA = 1.2
ENHANCE_BETA = 30

# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GRAY = (100, 100, 100)

# HUD Settings
HUD_BOX_START = (40, 40)
HUD_BOX_END = (220, 200)
COUNT_TEXT_POS = (105, 155)
LABEL_TEXT_POS = (45, 230)
FPS_TEXT_POS = (1050, 50)
