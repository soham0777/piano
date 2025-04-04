import os
import cv2
import time
import mediapipe as mp
import pyautogui
from flask import Flask, jsonify
from pyvirtualdisplay import Display

# 🖥️ Start a Virtual Display (since Render has no GUI)
display = Display(visible=0, size=(1024, 768))
display.start()

# Initialize Flask app
app = Flask(__name__)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)  # ❌ This won't work on Render (No real webcam)

def count_fingers(hand_landmarks):
    """Count the number of extended fingers"""
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    extended_fingers = sum(
        1 for tip in finger_tips if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y
    )
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
            key_mapping = {
                0: ("Do nothing", "space"),
                1: ("Jump", "up"),
                2: ("Slide", "down"),
                3: ("Turn Right", "right"),
                4: ("Turn Left", "left"),
            }
            action, key = key_mapping.get(extended_fingers, ("Unknown", None))

            if key:
                pyautogui.press(key)

    return jsonify({"action": action})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
