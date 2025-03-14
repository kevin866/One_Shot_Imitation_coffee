
# Step 3: Import necessary libraries
import os
import cv2
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
import numpy as np

# Step 4: Set paths
model_path = "best.pt"  # Update to the local model path
video_path = "v2.mp4"  # Update to the local video file path
output_path = "output_detection.mp4"  # Local output path


# Step 5: Define custom classes to look for in the YOLO model
custom_classes = ['coffee_maker', 'cup', 'mug', 'hand', 'coffee_filter', 'coffee_bag',
                 'coffee_container', 'spoon', 'kettle', 'coffee_grounds', 'water', 'person']

# Step 6: Load the YOLO model
model = YOLO(model_path)
print(f"Model loaded: {model_path}")

# Check available classes in model
class_names = model.names
print(f"Available classes in model: {class_names}")

# Check if 'hand' is in the model's classes, if not we'll look for 'person' as a fallback
has_hand_class = False
has_person_class = False
for idx, name in class_names.items():
    if name.lower() == 'hand':
        has_hand_class = True
    if name.lower() == 'person':
        has_person_class = True

if has_hand_class:
    print("Model can detect hands directly")
elif has_person_class:
    print("Model can detect people but not hands specifically. Will use person detection as proxy for hands")
else:
    print("Note: This model doesn't have specific classes for hands or people")

# Print which coffee-making items are already in the model
available_items = []
missing_items = []
for item in custom_classes:
    found = False
    for idx, name in class_names.items():
        if item.lower() in name.lower() or name.lower() in item.lower():
            available_items.append(f"{item} (as {name})")
            found = True
            break
    if not found:
        missing_items.append(item)

print(f"Coffee items already detectable by model: {available_items}")
print(f"Coffee items not in model (may not be detected): {missing_items}")

# Step 7: Process the video with standard YOLO detection for all objects
def process_video(input_video, output_video, model, conf_threshold=0.25, display_interval=2):
    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Process the video frame by frame
    start_time = time.time()
    frame_count = 0

    plt.figure(figsize=(12, 8))

    # For hand detection from upper limbs if no hand class in model
    def extract_hand_regions(boxes, class_ids, class_names):
        hand_boxes = []
        for i, cls_id in enumerate(class_ids):
            # If it's a person class and we have no direct hand class
            if class_names[int(cls_id)] == 'person' and not has_hand_class:
                x1, y1, x2, y2 = boxes[i]
                # Extract the upper part of the person detection as potential hand region
                # This is a very simple heuristic - upper 1/3 of the person
                hand_y = y1 + (y2 - y1) // 3
                hand_boxes.append([x1, y1, x2, hand_y, 'hand', 0.6])  # Add as hand with confidence 0.6
        return hand_boxes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame, conf=conf_threshold)

        # Get the annotated frame with YOLO detections
        annotated_frame = results[0].plot()

        # Extract hand regions from person detections if needed
        if not has_hand_class and has_person_class:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            hand_regions = extract_hand_regions(boxes, class_ids, model.names)

            # Draw hand regions on the frame
            for x1, y1, x2, y2, label, conf in hand_regions:
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Simple method to try to detect coffee filters (white circular objects)
        if 'coffee_filter' in missing_items:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create a mask for white objects
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Look for circular white contours that could be coffee filters
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter by size to avoid small noise
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.5:  # Filter by circularity (circle has value 1)
                            # Draw a bounding box
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = float(w) / h
                            # Filter reasonably circular objects
                            if 0.7 < aspect_ratio < 1.3:
                                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                                cv2.putText(annotated_frame, "Possible Filter", (x, y-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Write the frame to the output video
        out.write(annotated_frame)

        # Display the frame with detections at specified intervals
        frame_count += 1
        if frame_count % display_interval == 0:
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time
            remaining_frames = total_frames - frame_count
            estimated_time = remaining_frames / frames_per_second if frames_per_second > 0 else 0

            print(f"Processed {frame_count}/{total_frames} frames "
                  f"({frame_count/total_frames*100:.1f}%) "
                  f"- {frames_per_second:.1f} fps "
                  f"- Est. time remaining: {estimated_time/60:.1f} minutes")

            # Display the annotated frame
            plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"Frame {frame_count}/{total_frames} with Coffee Making Detections")
            plt.show()

            # Print detections
            print("Detections in this frame:")
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                print(f"  • {class_name}: {conf:.2f} confidence, bbox: [{x1}, {y1}, {x2}, {y2}]")

            print("-" * 50)

    # Release resources
    cap.release()
    out.release()

    print(f"Video processing completed in {time.time() - start_time:.2f} seconds.")
    print(f"Output saved to: {output_video}")

    return output_video

# Step 8: Enhanced detection settings
# Lower confidence threshold to catch more potential coffee items
conf_threshold = 0.2  # Lower threshold to detect more objects
display_interval = 2  # Show every 2nd frame as requested

# Run the inference with enhanced settings
output_file = process_video(video_path, output_path, model, conf_threshold=conf_threshold, display_interval=display_interval)

# Step 9: Display a sample frame from the final result
def display_sample_frame(video_path, frame_num=2):
    """Display a sample frame from the processed video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.title(f"Sample Frame #{frame_num} from Processed Video")
        plt.show()
    else:
        print(f"Could not extract frame #{frame_num} from the video.")

# Display a sample frame from the processed video
print("Final processed video sample:")
display_sample_frame(output_path)

print("Download the processed video:")

# Save the output directly to your Google Drive
drive_output_path = "/content/drive/MyDrive/detected_first.mp4"
