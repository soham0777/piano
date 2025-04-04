import os
import cv2
import time
import mediapipe as mp
import pyautogui
from flask import Flask, jsonify

# Virtual Display Setup (Required for Render)
os.environ["DISPLAY"] = ":99"
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()

# Initialize Flask app
app = Flask(__name__)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    """Count the number of extended fingers"""
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    extended_fingers = 0

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            extended_fingers += 1

    return extended_fingers

@app.route('/detect', methods=['GET'])
def detect_gesture():
    """Detect hand gestures and return action"""
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to read from webcam"}), 500

    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)

    # Convert to RGB and process with Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    action = "No Hand Detected"
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            extended_fingers = count_fingers(hand_landmarks)

            # Map gestures to actions
            if extended_fingers == 0:
                action = "Do nothing"
                pyautogui.press('space')
            elif extended_fingers == 1:
                action = "Jump"
                pyautogui.press('up')
            elif extended_fingers == 2:
                action = "Slide"
                pyautogui.press('down')
            elif extended_fingers == 3:
                action = "Turn Right"
                pyautogui.press('right')
            elif extended_fingers == 4:
                action = "Turn Left"
                pyautogui.press('left')

    return jsonify({"action": action})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
