import cv2
from typing import Tuple, Optional, Dict, Any

class GUIView:
    """
    View layer responsible for visual rendering, GUI overlay drawings, and camera window display.
    """
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.window_name = "Hand Gesture Volume Control (MVC)"

    def read_frame(self) -> Tuple[bool, Any]:
        """Read a frame from webcam and flip horizontally for intuitive selfie view."""
        success, img = self.cap.read()
        if success:
            img = cv2.flip(img, 1)
        return success, img

    def render_finger_landmarks(self, img, pinch_data: Dict[str, Any]) -> None:
        """
        Draw circles on landmark 4 and 8, line between them, and center point indicator.
        """
        x1, y1 = pinch_data["p1"]
        x2, y2 = pinch_data["p2"]
        cx, cy = pinch_data["center"]
        distance = pinch_data["distance"]

        # Draw outer landmark circles
        cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (255, 0, 255), cv2.FILLED)

        # Draw connecting line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Center point indicator - turns green when pinched close (< 25px)
        center_color = (0, 255, 0) if distance < 25 else (255, 0, 255)
        cv2.circle(img, (cx, cy), 10 if distance < 25 else 8, center_color, cv2.FILLED)

    def render_volume_hud(self, img, vol_bar: float, vol_per: float) -> None:
        """
        Render vertical Volume Bar GUI and percentage text HUD.
        """
        # Outer Volume Bar Box
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
        # Filled Volume Level
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
        # Text Overlay Percentage
        cv2.putText(img, f'{int(vol_per)} %', (40, 450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    def show_frame(self, img) -> bool:
        """
        Display image frame in OpenCV window and check for quit key ('q').
        Returns True if program should continue, False if user pressed 'q'.
        """
        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')


    def close(self) -> None:
        """Release camera hardware and close windows."""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
