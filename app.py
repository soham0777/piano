import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):                  
    #  Count the number of extended fingers
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    extended_fingers = 0

    # Check if each finger is extended
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            extended_fingers += 1

    return extended_fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            extended_fingers = count_fingers(hand_landmarks)
            print(f"Fingers extended: {extended_fingers}")  # Debugging output

            # Control the game based on the number of extended fingers
            if extended_fingers == 0:
                print("Do nothing")
                pyautogui.press('space')  # Simulate space key press
            elif extended_fingers == 1:
                print("Jump")
                pyautogui.press('up')  # Simulate up arrow key press
            elif extended_fingers == 2:
                print("Slide")
                pyautogui.press('down')  # Simulate down arrow key press
            elif extended_fingers == 3:
                print("Turn Right")
                pyautogui.press('right')  # Simulate right arrow key press
            elif extended_fingers == 4:
                print("Turn Left")
                pyautogui.press('left')  # Simulate left arrow key press   

            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow('Hand Gesture Control for Subway Surfers', frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(0.1)  # Small delay to prevent excessive CPU usage

# Release resources
cap.release()
cv2.destroyAllWindows()
