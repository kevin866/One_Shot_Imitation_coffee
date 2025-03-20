import cv2
import numpy as np
import time
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('models/best5_multi_label/weights/best.pt')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Video input/output paths
input_video_path = 'v3.MOV'
output_video_path = 'output_with_tracking.mp4'

# Set movement thresholds
movement_threshold = 80
glittering_threshold = 300

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Object and hand tracking storage
object_positions = {}  # Object tracking
hand_positions = {}  # Hand tracking
movement_events = []

# Function to calculate Euclidean distance
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Function to track movement (used for both objects and hands)
def track_movement(tracking_dict, obj_id, current_position, current_time):
    if obj_id not in tracking_dict:
        tracking_dict[obj_id] = (current_position, current_time)
        return

    last_position, last_time = tracking_dict[obj_id]
    movement_distance = calculate_distance(last_position, current_position)

    if movement_threshold < movement_distance < glittering_threshold:
        movement_events.append({
            "object_id": obj_id,
            "time": current_time,
            "from_position": last_position,
            "to_position": current_position,
            "distance": movement_distance
        })
        tracking_dict[obj_id] = (current_position, current_time)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    # Perform YOLO object detection
    results = model(frame)

    # Perform hand tracking
    hand_results = hands.process(rgb_frame)
    grasping_points = []
    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Extract thumb tip (landmark 4) and index tip (landmark 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Convert to pixel coordinates
            thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
            index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

            # Estimate grasping point (near the thumb tip or between thumb and index)
            grasping_x = int((thumb_x + index_x) / 2)  # Midpoint
            grasping_y = int((thumb_y + index_y) / 2)

            grasping_points.append((grasping_x, grasping_y))

            # Draw grasping point
            cv2.circle(frame, (grasping_x, grasping_y), 8, (0, 0, 255), -1)  # Red circle


            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Process detected objects
    for result in results:
        boxes = result.boxes
        labels = result.names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Bounding box coordinates
            conf = float(box.conf[0].cpu().numpy())  # Confidence score
            cls = int(box.cls[0].cpu().numpy())  # Class ID
            
            # Object center
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            center_position = (center_x, center_y)

            # Track object movement
            track_movement(object_positions, f"obj_{cls}", center_position, current_time)

            # Draw bounding box
            label = f"{labels[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw movement paths
    for event in movement_events:
        from_pos = tuple(map(int, event["from_position"]))
        to_pos = tuple(map(int, event["to_position"]))
        cv2.arrowedLine(frame, from_pos, to_pos, (0, 255, 255), 2)
        cv2.circle(frame, from_pos, 5, (255, 0, 0), -1)  # Start point
        cv2.circle(frame, to_pos, 5, (0, 255, 0), -1)  # End point

    # Write to output video
    out.write(frame)

    # Display frame
    cv2.imshow('Tracking Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
