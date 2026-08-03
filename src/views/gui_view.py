import cv2
from typing import Tuple, Optional, Dict, Any
from src import config
import mediapipe as mp

class GUIView:
    """
    View layer responsible for visual rendering, GUI overlay drawings, and camera window display.
    """
    def __init__(self, camera_index: int = config.CAMERA_INDEX, width: int = config.WINDOW_WIDTH, height: int = config.WINDOW_HEIGHT):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {camera_index}. Please check your webcam connection.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.window_name = config.WINDOW_NAME
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def read_frame(self) -> Tuple[bool, Any]:
        """Capture a frame from the webcam and flip it."""
        success, img = self.cap.read()
        if success:
            img = cv2.flip(img, 1)
        return success, img

    def render_hand_landmarks(self, img: Any, hand_data: Dict[str, Any]) -> None:
        """Draw the hand skeleton for the selected hand."""
        self.mp_draw.draw_landmarks(
            img,
            hand_data["hand_landmarks"],
            self.mp_hands.HAND_CONNECTIONS
        )

    @staticmethod
    def _hud_geometry(img: Any) -> Dict[str, Tuple[int, int]]:
        """Scale HUD geometry from the current frame size."""
        height, width = img.shape[:2]
        start_x = int(width * config.HUD_MARGIN_X_RATIO)
        start_y = int(height * config.HUD_MARGIN_Y_RATIO)
        box_width = int(width * config.HUD_BOX_WIDTH_RATIO)
        box_height = int(height * config.HUD_BOX_HEIGHT_RATIO)

        return {
            "box_start": (start_x, start_y),
            "box_end": (start_x + box_width, start_y + box_height),
            "count_pos": (start_x + int(box_width * 0.35), start_y + int(box_height * 0.72)),
            "label_pos": (start_x, start_y + box_height + int(height * 0.045)),
            "warning_pos": (int(width * 0.31), start_y + 10),
            "fps_pos": (width - int(width * config.FPS_MARGIN_X_RATIO), start_y + 10),
        }

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
        geometry = self._hud_geometry(img)

        # Outer HUD Box Background
        cv2.rectangle(img, geometry["box_start"], geometry["box_end"], config.COLOR_BLACK, cv2.FILLED)
        cv2.rectangle(img, geometry["box_start"], geometry["box_end"], config.COLOR_WHITE, 3)

        if hand_data:
            count = hand_data["total_count"]
            hand_label = hand_data["hand_label"]
            out_of_bounds = hand_data.get("out_of_bounds", False)
            
            # Display Count Number inside HUD Box
            cv2.putText(img, str(count), geometry["count_pos"],
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, config.COLOR_GREEN, 6)
            
            # Display Hand Label below HUD Box
            cv2.putText(img, f"Hand: {hand_label}", geometry["label_pos"],
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, config.COLOR_WHITE, 2)

            if out_of_bounds:
                cv2.putText(img, "WARNING: Hand Out of Bounds!", geometry["warning_pos"],
                            cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_RED, 3)
        else:
            # Display NO HAND detected state
            cv2.putText(img, "0", geometry["count_pos"],
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, config.COLOR_GRAY, 6)
            cv2.putText(img, "No Hand Detected", geometry["label_pos"],
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, config.COLOR_ORANGE, 2)

    def render_fps(self, img: Any, fps: int) -> None:
        """
        Render FPS counter on the top right of the screen.
        """
        geometry = self._hud_geometry(img)
        cv2.putText(img, f"FPS: {fps}", geometry["fps_pos"], cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_BLUE, 3)

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
