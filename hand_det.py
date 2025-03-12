import mediapipe as mp
import cv2
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Sensitivity Threshold (Higher = Less Sensitive)
CURL_THRESHOLD = 0.03  # Adjust this to fine-tune sensitivity
OPEN_THRESHOLD = -0.02
def is_finger_curled(landmarks, finger_indices):
    """
    Detects if a finger is curled based on its landmarks with a threshold.

    Parameters:
        landmarks (list): List of hand landmarks from MediaPipe.
        finger_indices (tuple): Indices of the (MCP, PIP, DIP, TIP) joints for the finger.

    Returns:
        bool: True if the finger is curled, False if it is open.
    """
    MCP, PIP, DIP, TIP = finger_indices

    # Y-values are used (lower values = higher on screen)
    tip_y, dip_y, pip_y, mcp_y = landmarks[TIP].y, landmarks[DIP].y, landmarks[PIP].y, landmarks[MCP].y

    # Finger is curled if the TIP is significantly lower than DIP & PIP
    if (tip_y > dip_y + CURL_THRESHOLD) and (tip_y > pip_y + CURL_THRESHOLD):
        return "curled"  # Finger is curled
    
    elif (tip_y < dip_y - OPEN_THRESHOLD) and (tip_y < pip_y - OPEN_THRESHOLD):
        return "open"
    return "buffer"  # Finger is open


def calculate_centroid(landmarks):
    """
    Calculate the centroid of the hand landmarks (average x, y).
    """
    x_coords = [landmarks[i].x for i in range(21)]
    y_coords = [landmarks[i].y for i in range(21)]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return centroid_x, centroid_y



# Finger indices according to MediaPipe
FINGER_INDICES = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
hand_state = "opened"
last_state_duration = time.time()
# Capture video
hand_events = {"close": [], "open": []}
cap = cv2.VideoCapture('v2.mp4')
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark positions
                landmarks = hand_landmarks.landmark
                
                # Check each finger's state
                finger_states = {}
                curled_fingers = 0
                opened_fingers = 0
                for finger, indices in FINGER_INDICES.items():
                    is_curled = is_finger_curled(landmarks, indices)
                    finger_states[finger] = is_curled
                    if is_curled == "curled":
                        curled_fingers += 1
                    elif is_curled == "open":
                        opened_fingers += 1
                
                if curled_fingers>=3 and hand_state!="holding" and time.time()-last_state_duration>1.0:
                    hand_state="holding"
                    last_state_duration = time.time()
                    centroid_x, centroid_y = calculate_centroid(landmarks)
                    hand_events["close"].append((centroid_x, centroid_y))
                elif opened_fingers>2 and hand_state!="opened" and time.time()-last_state_duration>1.0:
                    hand_state = "opened"
                    last_state_duration = time.time()
                    centroid_x, centroid_y = calculate_centroid(landmarks)
                    hand_events["open"].append((centroid_x, centroid_y))


                # Display the hand state
                cv2.putText(frame, f"Hand State: {hand_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Display each finger's state
                for i, (finger, curled) in enumerate(finger_states.items()):
                    text = f"{finger}: {curled}"
                    cv2.putText(frame, text, (10, 60 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Hand Gesture Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(hand_events)