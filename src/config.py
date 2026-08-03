from dataclasses import dataclass
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
LOW_LIGHT_THRESHOLD: int = 90
HIGH_LIGHT_THRESHOLD: int = 190
BRIGHT_SCENE_BETA: int = -10

# Finger Analysis
FINGER_ANGLE_THRESHOLD: float = 155.0
THUMB_ANGLE_THRESHOLD: float = 150.0
DISTANCE_MARGIN: float = 0.015

# Temporal Smoothing
SMOOTHING_WINDOW: int = 5

# Colors (BGR format for OpenCV)
COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)
COLOR_RED: Tuple[int, int, int] = (0, 0, 255)
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_BLACK: Tuple[int, int, int] = (0, 0, 0)
COLOR_ORANGE: Tuple[int, int, int] = (0, 165, 255)
COLOR_BLUE: Tuple[int, int, int] = (255, 0, 0)
COLOR_GRAY: Tuple[int, int, int] = (100, 100, 100)
COLOR_CYAN: Tuple[int, int, int] = (255, 255, 0)
COLOR_YELLOW: Tuple[int, int, int] = (0, 255, 255)

# HUD Settings
HUD_MARGIN_X_RATIO: float = 0.03
HUD_MARGIN_Y_RATIO: float = 0.05
HUD_BOX_WIDTH_RATIO: float = 0.14
HUD_BOX_HEIGHT_RATIO: float = 0.22
FPS_MARGIN_X_RATIO: float = 0.18
SIDE_PANEL_WIDTH_RATIO: float = 0.26
SIDE_PANEL_HEIGHT_RATIO: float = 0.28

# Runtime Display Defaults
SHOW_DIAGNOSTICS_BY_DEFAULT: bool = True
SHOW_CONTROLS_BY_DEFAULT: bool = True
SHOW_SKELETON_BY_DEFAULT: bool = True


@dataclass(frozen=True)
class AppSettings:
    """Runtime-configurable application settings."""
    camera_index: int = CAMERA_INDEX
    window_width: int = WINDOW_WIDTH
    window_height: int = WINDOW_HEIGHT
    max_hands: int = MAX_HANDS
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE
    smoothing_window: int = SMOOTHING_WINDOW
    show_diagnostics: bool = SHOW_DIAGNOSTICS_BY_DEFAULT
    show_controls: bool = SHOW_CONTROLS_BY_DEFAULT
    show_skeleton: bool = SHOW_SKELETON_BY_DEFAULT
