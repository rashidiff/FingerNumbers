"""Convert stable hand observations into application-friendly events."""

from dataclasses import dataclass
from time import monotonic
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GestureEvent:
    hand_label: str
    gesture: str
    event_type: str
    timestamp: float
    confidence: float


class GestureEngine:
    """Emit one change event and one optional hold event per gesture."""

    def __init__(self, hold_seconds: float = 1.0):
        self.hold_seconds = max(0.0, hold_seconds)
        self._states: Dict[str, Dict[str, object]] = {}

    def update(self, observation, timestamp: Optional[float] = None) -> List[GestureEvent]:
        now = monotonic() if timestamp is None else timestamp
        key = observation.hand_label
        gesture = observation.gesture_label
        state = self._states.get(key)
        if state is None or state["gesture"] != gesture:
            self._states[key] = {"gesture": gesture, "started_at": now, "held": False}
            return [GestureEvent(key, gesture, "changed", now, observation.confidence)]

        if not state["held"] and now - float(state["started_at"]) >= self.hold_seconds:
            state["held"] = True
            return [GestureEvent(key, gesture, "held", now, observation.confidence)]
        return []

    def reset(self) -> None:
        self._states.clear()
