import cv2
import os

# Open video file
video_path = "v3.MOV"  # Change this to your video path
cap = cv2.VideoCapture(video_path)

# Get frames per second (fps) and total frame count
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames // fps  # Duration in seconds

# Create output folder
output_folder = "image"
os.makedirs(output_folder, exist_ok=True)

# Extract frames
for sec in range(duration):
    frame_id = sec * fps  # Frame corresponding to each second
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    success, frame = cap.read()
    if success:
        filename = os.path.join(output_folder, f"frame_{sec}.jpg")
        cv2.imwrite(filename, frame)
    else:
        print(f"Failed to capture frame at {sec} seconds.")

cap.release()
print(f"Frames saved in '{output_folder}' folder.")
