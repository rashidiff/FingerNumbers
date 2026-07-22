import cv2
from typing import Tuple, Optional, Dict, Any
from src import config

class GUIView:
    """
    View layer responsible for visual rendering, GUI overlay drawings, and camera window display.
    """
    def __init__(self, camera_index: int = config.CAMERA_INDEX, width: int = config.WINDOW_WIDTH, height: int = config.WINDOW_HEIGHT):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.window_name = config.WINDOW_NAME

    def read_frame(self) -> Tuple[bool, Any]:
        """Capture a frame from the webcam and flip it."""
        success, img = self.cap.read()
        if success:
            img = cv2.flip(img, 1)
        return success, img

    def render_finger_highlights(self, img: Any, hand_data: Dict[str, Any]) -> None:
        """
        Draw visual indicators on the finger tips. Green for extended, Red for closed.
        """
        lm_list = hand_data["lm_list"]
        finger_states = hand_data["finger_states"]
        tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
        
        for idx, tip_id in enumerate(tip_ids):
            cx, cy = lm_list[tip_id]
            is_open = finger_states[idx] == 1
            color = config.COLOR_GREEN if is_open else config.COLOR_RED
            radius = 12 if is_open else 8
            cv2.circle(img, (cx, cy), radius, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), radius + 2, config.COLOR_WHITE, 2)

    def render_count_hud(self, img: Any, hand_data: Optional[Dict[str, Any]]) -> None:
        """
        Render visual HUD counter box on top-left of the screen.
        """
        # Outer HUD Box Background
        cv2.rectangle(img, config.HUD_BOX_START, config.HUD_BOX_END, config.COLOR_BLACK, cv2.FILLED)
        cv2.rectangle(img, config.HUD_BOX_START, config.HUD_BOX_END, config.COLOR_WHITE, 3)

        if hand_data:
            count = hand_data["total_count"]
            hand_label = hand_data["hand_label"]
            
            # Display Count Number inside HUD Box
            cv2.putText(img, str(count), config.COUNT_TEXT_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, config.COLOR_GREEN, 6)
            
            # Display Hand Label below HUD Box
            cv2.putText(img, f"Hand: {hand_label}", config.LABEL_TEXT_POS,
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, config.COLOR_WHITE, 2)
        else:
            # Display NO HAND detected state
            cv2.putText(img, "0", config.COUNT_TEXT_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, config.COLOR_GRAY, 6)
            cv2.putText(img, "No Hand Detected", config.LABEL_TEXT_POS,
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, config.COLOR_ORANGE, 2)

    def render_fps(self, img: Any, fps: int) -> None:
        """
        Render FPS counter on the top right of the screen.
        """
        cv2.putText(img, f"FPS: {fps}", config.FPS_TEXT_POS, cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_BLUE, 3)

    def show_frame(self, img: Any) -> bool:
        """
        Display rendered frame in OpenCV window and check for exit key ('q') or window close button.
        Returns True if continuing, False if 'q' pressed or window closed.
        """
        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1) & 0xFF
        
        # Check if the window was closed by the user clicking the 'X' button
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            # If the window is already destroyed, property check might throw an error
            return False

        return key != ord('q')

    def close(self) -> None:
        """Release webcam resource and destroy windows."""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
