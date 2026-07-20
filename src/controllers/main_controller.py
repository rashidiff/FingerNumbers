import cv2
import numpy as np
from src.models.audio_model import AudioModel
from src.models.hand_detector_model import HandDetectorModel
from src.views.gui_view import GUIView

class MainController:
    """
    Controller layer coordinating user input from Vision Model (Hand Tracking),
    system action in Audio Model, and visual updates in View.
    """
    def __init__(self, min_dist: float = 20.0, max_dist: float = 200.0):
        self.audio_model = AudioModel()
        self.hand_model = HandDetectorModel(max_hands=1)
        self.view = GUIView()

        # Calibration parameters for pinch distance mapping (in pixels)
        self.min_dist = min_dist
        self.max_dist = max_dist

        # State tracking for GUI rendering
        self.vol_bar = 400.0
        self.vol_per = 0.0

    def run(self) -> None:
        """Main execution loop for application."""
        print("Application Started. Press 'q' in the window to quit.")
        try:
            while True:
                success, img = self.view.read_frame()
                if not success:
                    print("Error: Could not retrieve camera frame.")
                    break

                # Step 1: Convert frame to RGB for MediaPipe processing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hand_model.process_frame(img_rgb)

                # Step 2: Extract hand pinch landmarks if hand is detected
                pinch_data = self.hand_model.extract_pinch_landmarks(img, results)

                if pinch_data:
                    distance = pinch_data["distance"]

                    # Step 3: Interpolate distance range [min_dist, max_dist] to:
                    # - Decibels for Pycaw hardware Audio control
                    # - Vertical pixel position for Volume Bar (400 to 150)
                    # - Percentage (0 to 100)
                    min_vol = self.audio_model.get_min_vol()
                    max_vol = self.audio_model.get_max_vol()

                    vol_db = np.interp(distance, [self.min_dist, self.max_dist], [min_vol, max_vol])
                    self.vol_bar = np.interp(distance, [self.min_dist, self.max_dist], [400, 150])
                    self.vol_per = np.interp(distance, [self.min_dist, self.max_dist], [0, 100])

                    # Update system audio volume
                    self.audio_model.set_volume_db(vol_db)

                    # Draw landmarks overlay
                    self.view.render_finger_landmarks(img, pinch_data)

                # Step 4: Render UI HUD (Volume Bar & Text)
                self.view.render_volume_hud(img, self.vol_bar, self.vol_per)

                # Step 5: Render frame and check exit input
                if not self.view.show_frame(img):
                    break
        finally:
            self.view.close()
            print("Application terminated cleanly.")
