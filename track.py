import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv5 model
model = YOLO('models/best5_multi_label/weights/best.pt')

# Video input/output paths
input_video_path = 'v3.MOV'
output_video_path = 'output_with_tracking.mp4'

# Set movement threshold for hysteresis
movement_threshold = 80  # Adjust based on your object's size and expected movement
glitering_threshold = 300
# Initialize video capture
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# For storing the last positions of objects (by object_id)
object_positions = {}  # object_id -> (last_position, last_time)

# For storing movement events with time
movement_events = []

# Initialize object tracking IDs (use unique ID for each detected object)
next_object_id = 0

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

# Start processing the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the cropped frame
    results = model(frame)  # Make predictions


    # # Perform detection on the frame using YOLO
    # results = model(frame)  # Make predictions

    # Extract bounding boxes and labels
    for result in results:
        boxes = result.boxes  # Extract the boxes from the result object
        labels = result.names  # Get class labels

        current_time = time.time()

        # Loop through each detected object
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
            conf = box.conf[0].cpu().numpy()  # Get confidence score
            cls = box.cls[0].cpu().numpy()  # Get class ID
            # print(conf)
            # Generate object_id based on class and bounding box location
            obj_id = int(cls)  # Use the class ID as the object ID

            # Calculate the center of the bounding box for movement detection
            center_position = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Track movement for the object
            track_object(obj_id, center_position, current_time)

            # Draw bounding box and object info
            label = f"{labels[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Annotate movement paths (arrows or lines) for the tracked objects
        for event in movement_events:
            object_id = event['object_id']
            from_position = event['from_position']
            to_position = event['to_position']
            time1 = event['current_time']
            
            # Ensure to_position is a tuple of integers
            to_position = (int(to_position[0]), int(to_position[1]))
            from_position = (int(from_position[0]), int(from_position[1]))

            # Draw arrow showing movement
            cv2.arrowedLine(frame, from_position, to_position, (0, 255, 255), 2)

            # Draw the last position (from_position) as a circle (in blue)
            cv2.circle(frame, from_position, 5, (255, 0, 0), -1)  # Blue circle

            # Draw the current position (to_position) as a circle (in green)
            cv2.circle(frame, to_position, 5, (0, 255, 0), -1)  # Green circle

            # Optionally, annotate the object ID and time
            cv2.putText(frame, f"ID: {object_id} Time: {time1:.2f}", 
                (to_position[0], to_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the current frame (optional)
    cv2.imshow('Tracking Output', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

import json

# After processing the video, save the movement events to a JSON file
with open('movement_events.json', 'w') as json_file:
    json.dump(movement_events, json_file, indent=4)

print("Movement events saved to movement_events.json")

    
    
