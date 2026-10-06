"""Typed domain objects shared by the vision and presentation layers."""

from dataclasses import dataclass
from typing import Any, List, Tuple


Point2D = Tuple[int, int]
Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class HandObservation:
    """One analyzed hand in a frame.

    The MediaPipe landmark object is intentionally kept here so renderers can
    draw it without coupling the controller to MediaPipe internals.
    """

    hand_landmarks: Any
    landmarks: List[Point2D]
    normalized_landmarks: List[Point3D]
    finger_states: List[int]
    total_count: int
    hand_label: str
    gesture_label: str
    stability: float
    bounding_box: BoundingBox
    out_of_bounds: bool
    confidence: float = 1.0

    # Compatibility bridge for existing renderers while the application
    # migrates from untyped dictionaries to domain objects.
    def __getitem__(self, key: str) -> Any:
        aliases = {"lm_list": "landmarks", "bounding_box": "bounding_box"}
        value = getattr(self, aliases.get(key, key))
        if key == "bounding_box":
            return {"x": value.x, "y": value.y, "width": value.width, "height": value.height}
        if key == "tip_ids":
            return [4, 8, 12, 16, 20]
        return value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default

    @property
    def center(self) -> Point2D:
        return (
            self.bounding_box.x + self.bounding_box.width // 2,
            self.bounding_box.y + self.bounding_box.height // 2,
        )

