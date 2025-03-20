import cv2
import numpy as np
import time
import mediapipe as mp
from ultralytics import YOLO
from collections import defaultdict


def generate_object_action(data):

    prev = 99
    start_time = 0
    ori_pos = 0
    fin_tim = 0
    fin_pos = 0
    # state = 0 no object in motion, state = 1 object in motion
    state = 0
    combined_events=[]
    for event in data:
        obj_id = event['object_id']
        from_position = event['from_position']
        to_position = event['to_position']
        last_time = event['last_time']
        if prev!=obj_id:
            if state != 1:
                start_time = last_time
                prev = obj_id
                ori_pos = from_position
                state = 1
              
            elif state ==1:                
                # Combine the data
                start_times.append(start_time)
                end_times.append(fin_tim)
                combined_event = {
                    'object_id': prev,
                    'start_time': start_time,
                    'finish_time': fin_tim,
                    'from_position': ori_pos,
                    'to_position': fin_pos,
                }

                start_time = last_time
                prev = obj_id
                ori_pos = from_position
                # Add the combined event to the result list
                combined_events.append(combined_event)
                start_time = last_time
        fin_tim = event['current_time']
        fin_pos = event['to_position']
    return combined_events

# Function to calculate Euclidean distance
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to handle object movement detection and hysteresis
def track_object(obj_id, current_position, current_time):
    global object_positions, movement_events

    if obj_id not in object_positions:
        # If it's a new object, just record its first position
        object_positions[obj_id] = (current_position, current_time)
        return

    last_position, last_time = object_positions[obj_id]

    # Calculate the distance moved between frames
    movement_distance = calculate_distance(last_position, current_position)

    
    if movement_distance > movement_threshold and movement_distance<glitering_threshold:
        movement_events.append({
            "object_id": obj_id,
            "last_time":last_time,
            "current_time": current_time,
            "from_position": last_position,
            "to_position": current_position,
            "distance":movement_distance
        })

        # Update the last position and time for the object
        object_positions[obj_id] = (current_position, current_time)
    elif movement_distance<10:
        object_positions[obj_id] = (current_position, current_time)
    return obj_id

# Load YOLOv5 model
model = YOLO('models/best5_multi_label/weights/best.pt')
prev_obj = 99
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Video input/output paths
input_video_path = 'v3.MOV'
output_video_path = 'output_with_tracking.mp4'

movement_threshold = 80  # Adjust based on your object's size and expected movement
glitering_threshold = 300

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Object and hand tracking storage
# Create a defaultdict where the default value is an empty list
object_position_history = defaultdict(list)
hand_positions = {}  # Hand tracking
movement_events = []
object_positions={}
grasping_points_history = defaultdict(list)
start_times = []
end_times = []
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second

frame_count = 0  # Start frame index from 0
stime = []
# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = round(frame_count / fps, 4)  # Time in seconds since video start
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform YOLO object detection
    results = model(frame)

    # Perform hand tracking
    hand_results = hands.process(rgb_frame)
    thumb_pos = 0
    if hand_results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Extract thumb tip (landmark 4) and index tip (landmark 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Convert to pixel coordinates
            thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
            index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
            thumb_pos = (thumb_x, thumb_y)
            # Estimate grasping point (near the thumb tip or between thumb and index)
            grasping_x = int((thumb_x + index_x) / 2)  # Midpoint
            grasping_y = int((thumb_y + index_y) / 2)
            grasping_points_history[str(current_time)].append((grasping_x, grasping_y))
  
    # Extract bounding boxes and labels
    for result in results:
        boxes = result.boxes  # Extract the boxes from the result object
        labels = result.names  # Get class labels


        # Loop through each detected object
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
            conf = box.conf[0].cpu().numpy()  # Get confidence score
            cls = box.cls[0].cpu().numpy()  # Get class ID
            # Generate object_id based on class and bounding box location
            obj_id = int(cls)  # Use the class ID as the object ID

            # Calculate the center of the bounding box for movement detection
            center_position = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Track movement for the object
            track_object(obj_id, center_position, current_time)
            object_position_history[str(current_time)].append([obj_id, center_position])

            if obj_id == 1 and thumb_pos!=0:
                # we are assuming that the user is using thumb to press the start button
                start_button_threshold = calculate_distance(thumb_pos, center_position)
                if start_button_threshold<30:
                    stime.append(current_time)
    frame_count+=1

# Release resources
cap.release()
# out.release()
combined_events = generate_object_action(movement_events)

import json

# Convert to JSON-compatible format
data = {
    "combined_events": combined_events,
    "object_position_history": object_position_history,
    "grasping_points_history":grasping_points_history,
    "stime":stime
}

# Save as JSON file
with open("data.json", "w") as f:
    json.dump(data, f)

















cv2.destroyAllWindows()



