import cv2
import time
from collections import deque
from src.models.finger_counter_model import FingerCounterModel
from src.views.gui_view import GUIView
from src import config

class MainController:
    """
    Controller layer coordinating user input from Computer Vision model (Finger Counter)
    and rendering on-screen updates in View.
    """
    def __init__(self):
        self.finger_model = FingerCounterModel()
        self.view = GUIView()
        self.finger_history = deque(maxlen=config.SMOOTHING_WINDOW)

    def _stabilize_hand_data(self, hand_data):
        """Apply a majority vote over recent frames to reduce flicker."""
        if not hand_data:
            self.finger_history.clear()
            return None

        self.finger_history.append(hand_data["finger_states"])
        smoothed_states = []

        for finger_idx in range(len(hand_data["finger_states"])):
            open_votes = sum(frame[finger_idx] for frame in self.finger_history)
            smoothed_states.append(1 if open_votes >= (len(self.finger_history) / 2) else 0)

        stabilized = dict(hand_data)
        stabilized["finger_states"] = smoothed_states
        stabilized["total_count"] = sum(smoothed_states)
        return stabilized

    def run(self) -> None:
        """Main execution loop for Finger Counter application."""
        print("Finger Counter Application Started. Press 'q' in the window to quit.")
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
                hand_data = self.finger_model.analyze_hand(img, results)
                hand_data = self._stabilize_hand_data(hand_data)

                # Step 3: Render finger highlights if hand is present
                if hand_data:
                    self.view.render_hand_landmarks(img, hand_data)
                    self.view.render_finger_highlights(img, hand_data)

                # Step 4: Render Finger Count HUD Box and FPS on screen
                self.view.render_count_hud(img, hand_data)
                self.view.render_fps(img, fps)

                # Step 5: Display rendered frame and check exit key
                if not self.view.show_frame(img):
                    break
        finally:
            self.finger_model.close()
            self.view.close()
            print("Application terminated cleanly.")
