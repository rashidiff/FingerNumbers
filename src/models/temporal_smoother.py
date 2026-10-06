"""Temporal filtering primitives for noisy per-frame finger predictions."""

from collections import deque
from typing import Deque, List, Optional


class FingerStateSmoother:
    """Apply majority filtering with hysteresis for each finger."""

    def __init__(self, window: int = 5, switch_ratio: float = 0.6):
        self.history: Deque[List[int]] = deque(maxlen=max(1, window))
        self.switch_ratio = min(1.0, max(0.5, switch_ratio))
        self._last_state: Optional[List[int]] = None

    def update(self, states: List[int]) -> List[int]:
        if not states:
            return []
        self.history.append(list(states))
        if self._last_state is None:
            self._last_state = list(states)
        frame_count = len(self.history)
        result = list(self._last_state)
        for index in range(len(states)):
            open_ratio = sum(frame[index] for frame in self.history) / frame_count
            if open_ratio >= self.switch_ratio:
                result[index] = 1
            elif open_ratio <= 1.0 - self.switch_ratio:
                result[index] = 0
        self._last_state = result
        return list(result)

    def reset(self) -> None:
        self.history.clear()
        self._last_state = None
