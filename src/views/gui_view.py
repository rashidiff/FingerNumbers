import cv2
from typing import Tuple, Dict, Any, Optional

class GUIView:
    """
    View layer responsible for camera frame capture, landmark overlays,
    and rendering Finger Count HUD on screen.
    """
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.window_name = "Hand Finger Counter (MVC)"

    def read_frame(self) -> Tuple[bool, Any]:
        """Read frame from camera and flip horizontally for selfie view."""
        success, img = self.cap.read()
        if success:
            img = cv2.flip(img, 1)
        return success, img

    def render_finger_highlights(self, img: Any, hand_data: Dict[str, Any]) -> None:
        """
        Highlight finger tips: Green circle for extended fingers, Magenta for closed fingers.
        """
        lm_list = hand_data["lm_list"]
        finger_states = hand_data["finger_states"]
        tip_ids = hand_data["tip_ids"]

        for idx, tip_id in enumerate(tip_ids):
            cx, cy = lm_list[tip_id]
            is_open = finger_states[idx] == 1
            color = (0, 255, 0) if is_open else (0, 0, 255)
            radius = 12 if is_open else 8
            cv2.circle(img, (cx, cy), radius, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), radius + 2, (255, 255, 255), 2)

    def render_count_hud(self, img: Any, hand_data: Optional[Dict[str, Any]]) -> None:
        """
        Render visual HUD counter box on top-left of the screen.
        """
        # Outer HUD Box Background
        cv2.rectangle(img, (40, 40), (220, 200), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (40, 40), (220, 200), (255, 255, 255), 3)

        if hand_data:
            count = hand_data["total_count"]
            hand_label = hand_data["hand_label"]
            
            # Display Count Number inside HUD Box
            cv2.putText(img, str(count), (105, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 0), 6)
            
            # Display Hand Label below HUD Box
            cv2.putText(img, f"Hand: {hand_label}", (45, 230),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Display NO HAND detected state
            cv2.putText(img, "0", (105, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, (100, 100, 100), 6)
            cv2.putText(img, "No Hand Detected", (45, 230),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 165, 255), 2)

    def show_frame(self, img: Any) -> bool:
        """
        Display rendered frame in OpenCV window and check for exit key ('q').
        Returns True if continuing, False if 'q' pressed.
        """
        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')

    def close(self) -> None:
        """Release webcam resource and destroy windows."""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
