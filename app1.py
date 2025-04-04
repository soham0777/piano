from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    extended_fingers = 0
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            extended_fingers += 1
    return extended_fingers

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(rgb_frame)
    action = "No Hand Detected"
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            extended_fingers = count_fingers(hand_landmarks)

            if extended_fingers == 0:
                action = "Do nothing"
            elif extended_fingers == 1:
                action = "Jump"
            elif extended_fingers == 2:
                action = "Slide"
            elif extended_fingers == 3:
                action = "Turn Right"
            elif extended_fingers == 4:
                action = "Turn Left"
    
    return jsonify({"action": action})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
