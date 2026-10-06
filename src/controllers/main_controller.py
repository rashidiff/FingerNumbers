import cv2
import time
from collections import deque
from dataclasses import replace
from typing import Any, Dict, Optional
from src.models.finger_counter_model import FingerCounterModel
from src.views.gui_view import GUIView
from src import config
from src.config import AppSettings
from src.models.types import HandObservation

class MainController:
    """
    Controller layer coordinating user input from Computer Vision model (Finger Counter)
    and rendering on-screen updates in View.
    """
    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or AppSettings()
        self.finger_model = FingerCounterModel(self.settings)
        self.view = GUIView(self.settings)
        self.finger_history = deque(maxlen=self.settings.smoothing_window)
        self.hand_histories = {}
        self.show_diagnostics = self.settings.show_diagnostics
        self.show_controls = self.settings.show_controls
        self.show_skeleton = self.settings.show_skeleton
        self.session_stats = {
            "frames_processed": 0,
            "fps_total": 0.0,
            "average_fps": 0.0,
            "peak_fps": 0,
            "count_frequency": {count: 0 for count in range(6)},
            "dominant_count": 0,
        }

    def _stabilize_hand_data(self, hand_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Apply a majority vote over recent frames to reduce flicker."""
        if not hand_data:
            self.finger_history.clear()
            return None

        self.finger_history.append(hand_data["finger_states"])
        smoothed_states = []

        for finger_idx in range(len(hand_data["finger_states"])):
            open_votes = sum(frame[finger_idx] for frame in self.finger_history)
            smoothed_states.append(1 if open_votes >= (len(self.finger_history) / 2) else 0)

        if isinstance(hand_data, HandObservation):
            stabilized = replace(hand_data)
        else:
            stabilized = dict(hand_data)
        if isinstance(stabilized, HandObservation):
            stabilized.finger_states = smoothed_states
            stabilized.total_count = sum(smoothed_states)
            stabilized.gesture_label = FingerCounterModel._describe_gesture(smoothed_states)
        else:
            stabilized["finger_states"] = smoothed_states
            stabilized["total_count"] = sum(smoothed_states)
            stabilized["gesture_label"] = FingerCounterModel._describe_gesture(smoothed_states)
        return stabilized

    def _stabilize_hands(self, observations):
        """Smooth each visible hand independently instead of mixing hands."""
        stabilized = []
        for observation in observations:
            key = observation.hand_label
            history = self.hand_histories.setdefault(key, deque(maxlen=self.settings.smoothing_window))
            previous = self.finger_history
            self.finger_history = history
            stabilized.append(self._stabilize_hand_data(observation))
            self.hand_histories[key] = self.finger_history
            self.finger_history = previous
        return stabilized

    def _update_session_stats(self, fps: int, hand_data: Optional[Dict[str, Any]]) -> None:
        """Keep lightweight per-session metrics for diagnostics."""
        self.session_stats["frames_processed"] += 1
        self.session_stats["fps_total"] += fps
        self.session_stats["average_fps"] = (
            self.session_stats["fps_total"] / self.session_stats["frames_processed"]
        )
        self.session_stats["peak_fps"] = max(self.session_stats["peak_fps"], fps)

        if hand_data:
            count = hand_data["total_count"]
            self.session_stats["count_frequency"][count] += 1
            self.session_stats["dominant_count"] = max(
                self.session_stats["count_frequency"],
                key=self.session_stats["count_frequency"].get
            )

    def _reset_session_stats(self) -> None:
        """Reset diagnostics counters without restarting the app."""
        self.session_stats = {
            "frames_processed": 0,
            "fps_total": 0.0,
            "average_fps": 0.0,
            "peak_fps": 0,
            "count_frequency": {count: 0 for count in range(6)},
            "dominant_count": 0,
        }

    def _handle_keypress(self, key: Optional[int]) -> bool:
        """Handle runtime key bindings and return False when the loop should stop."""
        if key is None or key == ord("q"):
            return False

        if key == ord("d"):
            self.show_diagnostics = not self.show_diagnostics
        elif key == ord("s"):
            self.show_skeleton = not self.show_skeleton
        elif key == ord("h"):
            self.show_controls = not self.show_controls
        elif key == ord("r"):
            self._reset_session_stats()
            self.finger_history.clear()
            self.hand_histories.clear()

        return True

    def run(self) -> None:
        """Main execution loop for Finger Counter application."""
        print("Finger Counter Application Started. Keys: q=quit d=diagnostics s=skeleton h=help r=reset.")
        pTime = 0
        try:
            while True:
                success, img = self.view.read_frame()
                if not success:
                    print("Error: Could not retrieve camera frame.")
                    break

                # Calculate FPS
                cTime = time.time()
                fps = int(1 / (cTime - pTime)) if pTime != 0 else 0
                pTime = cTime

                # Step 1: Convert frame to RGB for MediaPipe processing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.finger_model.process_frame(img_rgb)

                # Step 2: Analyze hand landmarks & count extended fingers
                hand_data_list = self._stabilize_hands(self.finger_model.analyze_hands(img, results))
                hand_data = hand_data_list[0] if hand_data_list else None
                self._update_session_stats(fps, hand_data)

                # Step 3: Render finger highlights if hand is present
                for observation in hand_data_list:
                    if self.show_skeleton:
                        self.view.render_hand_landmarks(img, observation)
                    self.view.render_hand_box(img, observation)
                    self.view.render_finger_highlights(img, observation)

                # Step 4: Render Finger Count HUD Box and FPS on screen
                self.view.render_count_hud(img, hand_data)
                self.view.render_fps(img, fps)
                if self.show_diagnostics:
                    self.view.render_diagnostics(img, hand_data, self.session_stats)
                if self.show_controls:
                    self.view.render_controls_hint(img)

                # Step 5: Display rendered frame and check exit key
                key = self.view.show_frame(img)
                if not self._handle_keypress(key):
                    break
        finally:
            self.finger_model.close()
            self.view.close()
            print("Application terminated cleanly.")
