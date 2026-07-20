import cv2
from src.models.finger_counter_model import FingerCounterModel
from src.views.gui_view import GUIView

class MainController:
    """
    Controller layer coordinating user input from Computer Vision model (Finger Counter)
    and rendering on-screen updates in View.
    """
    def __init__(self, max_hands: int = 1, detection_con: float = 0.5):
        self.finger_model = FingerCounterModel(max_hands=max_hands, detection_con=detection_con)
        self.view = GUIView()

    def run(self) -> None:
        """Main execution loop for Finger Counter application."""
        print("Finger Counter Application Started. Press 'q' in the window to quit.")
        try:
            while True:
                success, img = self.view.read_frame()
                if not success:
                    print("Error: Could not retrieve camera frame.")
                    break

                # Step 1: Convert frame to RGB for MediaPipe processing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.finger_model.process_frame(img_rgb)

                # Step 2: Analyze hand landmarks & count extended fingers
                hand_data = self.finger_model.analyze_hand(img, results)

                # Step 3: Render finger highlights if hand is present
                if hand_data:
                    self.view.render_finger_highlights(img, hand_data)

                # Step 4: Render Finger Count HUD Box on screen
                self.view.render_count_hud(img, hand_data)

                # Step 5: Display rendered frame and check exit key
                if not self.view.show_frame(img):
                    break
        finally:
            self.view.close()
            print("Application terminated cleanly.")
