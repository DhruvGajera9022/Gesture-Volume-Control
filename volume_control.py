import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
from collections import deque


class HandVolumeControl:
    def __init__(self, max_num_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        # Initialize webcam with error handling
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            raise Exception("Could not open webcam")

        # Set webcam resolution for better performance
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize Mediapipe Hands with better parameters
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.drawing_utils = mp.solutions.drawing_utils

        # Volume control parameters
        self.volume_delay = 0.3  # Reduced delay for better responsiveness
        self.last_volume_change = time.time()
        self.min_distance = 20  # Minimum distance threshold
        self.max_distance = 120  # Maximum distance threshold

        # Smoothing parameters
        self.distance_history = deque(maxlen=5)  # Store last 5 distances for smoothing

        # Visual feedback
        self.volume_level = 50  # Simulated volume level (0-100)
        self.volume_display_time = 0
        self.volume_display_duration = 1.0  # seconds

        # FPS calculation
        self.prev_time = 0

        # Status tracking
        self.hand_detected = False
        self.gesture_active = False


    """Apply smoothing to distance measurements to reduce jitter"""
    def smooth_distance(self, distance):
        self.distance_history.append(distance)
        return np.mean(self.distance_history)


    """Map distance to volume level (0-100)"""
    def map_distance_to_volume(self, distance):
        # Clamp distance to our range
        distance = max(self.min_distance, min(distance, self.max_distance))
        # Map to 0-100 range
        volume = ((distance - self.min_distance) / (self.max_distance - self.min_distance)) * 100
        return int(volume)


    """Draw a volume bar on the frame"""
    def draw_volume_bar(self, frame, volume_level):
        bar_x, bar_y = 50, 100
        bar_width, bar_height = 20, 200

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Volume level bar
        fill_height = int((volume_level / 100) * bar_height)
        color = (0, 255, 0) if volume_level > 30 else (0, 255, 255) if volume_level > 10 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y + bar_height - fill_height),
                      (bar_x + bar_width, bar_y + bar_height), color, -1)

        # Volume text
        cv2.putText(frame, f'Vol: {volume_level}%', (bar_x + 30, bar_y + bar_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    """Draw gesture information on the frame"""
    def draw_gesture_info(self, frame, distance, gesture_active):
        if gesture_active:
            # Draw distance
            cv2.putText(frame, f'Distance: {int(distance)}px', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw instructions
            cv2.putText(frame, 'Pinch: Volume Down | Spread: Volume Up', (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, 'Show your hand to control volume', (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    """Control system volume based on distance"""
    def control_volume(self, distance):
        current_time = time.time()

        if current_time - self.last_volume_change > self.volume_delay:
            volume_changed = False

            if distance > 80:  # Spread fingers - volume up
                pyautogui.press("volumeup")
                self.volume_level = min(100, self.volume_level + 10)
                volume_changed = True
            elif distance < 40:  # Pinch fingers - volume down
                pyautogui.press("volumedown")
                self.volume_level = max(0, self.volume_level - 10)
                volume_changed = True

            if volume_changed:
                self.last_volume_change = current_time
                self.volume_display_time = current_time


    """Process hand landmarks and control volume"""
    def process_hand_landmarks(self, frame, hand_landmarks):
        frame_height, frame_width = frame.shape[:2]

        # Get coordinates of thumb tip (id=4) and index fingertip (id=8)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        x1, y1 = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
        x2, y2 = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

        # Calculate distance
        distance = math.hypot(x2 - x1, y2 - y1)
        smoothed_distance = self.smooth_distance(distance)

        # Visual feedback
        # Draw circles on fingertips
        cv2.circle(frame, (x1, y1), 10, (0, 255, 255), -1)
        cv2.circle(frame, (x2, y2), 10, (0, 255, 255), -1)

        # Draw connecting line with color based on distance
        line_color = (0, 255, 0) if distance > 60 else (0, 255, 255) if distance > 40 else (0, 0, 255)
        cv2.line(frame, (x1, y1), (x2, y2), line_color, 3)

        # Draw center point
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)

        # Control volume
        self.control_volume(smoothed_distance)

        return smoothed_distance


    """Main loop for hand volume control"""
    def run(self):
        try:
            print("Hand Volume Control Started!")
            print("Controls:")
            print("- Pinch thumb and index finger: Volume Down")
            print("- Spread thumb and index finger: Volume Up")
            print("- Press ESC to quit")

            while True:
                success, frame = self.webcam.read()
                if not success:
                    print("Failed to read from webcam")
                    break

                # Flip frame horizontally for mirror effect
                # frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]

                # Convert BGR to RGB for Mediapipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands_detector.process(rgb_frame)

                # Reset status
                self.hand_detected = False
                self.gesture_active = False
                distance = 0

                # Process hand landmarks
                if results.multi_hand_landmarks:
                    self.hand_detected = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )

                        # Process landmarks and get distance
                        distance = self.process_hand_landmarks(frame, hand_landmarks)
                        self.gesture_active = True

                # Draw UI elements
                self.draw_volume_bar(frame, self.volume_level)
                self.draw_gesture_info(frame, distance, self.gesture_active)

                # Calculate and display FPS
                curr_time = time.time()
                fps = 1 / (curr_time - self.prev_time + 1e-5)
                self.prev_time = curr_time
                cv2.putText(frame, f'FPS: {int(fps)}', (frame_width - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2)

                # Display frame
                cv2.imshow("Hand Volume Control - Enhanced", frame)

                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == ord('r'):  # Reset volume level
                    self.volume_level = 50
                    print("Volume level reset to 50%")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()


    """Clean up resources"""
    def cleanup(self):
        print("Cleaning up...")
        self.webcam.release()
        cv2.destroyAllWindows()


# Usage
if __name__ == "__main__":
    try:
        # Disable pyautogui failsafe for better user experience
        pyautogui.FAILSAFE = True  # Keep failsafe enabled for safety

        # Create and run the hand volume controller
        controller = HandVolumeControl()
        controller.run()

    except Exception as e:
        print(f"Failed to start application: {e}")