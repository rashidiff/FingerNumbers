import cv2
import mediapipe as mp
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    # Initialize Webcam (0 is default camera)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize MediaPipe Hands solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,            # Detect only one hand as requested
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    # Initialize Pycaw for Windows Audio Endpoint Control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)

    # Get system volume range in dB (typically -65.25 dB to 0.0 dB)
    vol_range = volume.GetVolumeRange()
    min_vol = vol_range[0]
    max_vol = vol_range[1]

    # Default initial UI values
    vol_bar = 400
    vol_per = 0

    print("Gesture Volume Control started. Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab camera frame.")
            break

        # Flip image horizontally for intuitive selfie view
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        # Convert BGR image to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Optionally draw full hand skeletal landmarks
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract Landmark 4 (Thumb Tip) and Landmark 8 (Index Finger Tip)
                lm_thumb = hand_landmarks.landmark[4]
                lm_index = hand_landmarks.landmark[8]

                # Convert normalized coordinates (0.0 - 1.0) to pixel coordinates
                x1, y1 = int(lm_thumb.x * w), int(lm_thumb.y * h)
                x2, y2 = int(lm_index.x * w), int(lm_index.y * h)

                # Calculate Midpoint between Thumb Tip and Index Tip
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw circles on Thumb tip and Index tip
                cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 12, (255, 0, 255), cv2.FILLED)

                # Draw connecting line between the two tips
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Draw center point circle
                cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

                # Calculate Euclidean Distance between Thumb and Index tips
                distance = math.hypot(x2 - x1, y2 - y1)

                # Visual indicator when fingers are squeezed together (distance < 25)
                if distance < 25:
                    cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

                # Map finger distance (e.g., 20px to 200px) to:
                # 1. System volume range in dB (min_vol to max_vol)
                # 2. Volume bar vertical position on GUI (400px to 150px)
                # 3. Volume percentage text (0% to 100%)
                vol = np.interp(distance, [20, 200], [min_vol, max_vol])
                vol_bar = np.interp(distance, [20, 200], [400, 150])
                vol_per = np.interp(distance, [20, 200], [0, 100])

                # Update system master volume level
                volume.SetMasterVolumeLevel(vol, None)

        # Draw Graphic UI: Volume Bar
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)

        # Display Volume Percentage Text
        cv2.putText(img, f'{int(vol_per)} %', (40, 450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Display camera feed
        cv2.imshow("Hand Gesture Volume Control", img)

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
