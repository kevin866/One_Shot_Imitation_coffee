import json
import cv2
import numpy as np

# Load JSON file
with open("data.json", "r") as f:
    data = json.load(f)

combined_events = data["combined_events"]
object_position_history = data["object_position_history"]
grasping_points_history = data["grasping_points_history"]
stime = data['stime']
input_video_path = 'v3.MOV'
output_video_path = 'output.mp4'

start_times = [round(event['start_time'], 4) for event in combined_events]
end_times = [round(event['finish_time'], 4) for event in combined_events]

cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

pre_actions=["Placing paper filter", 
             "paper filter placed",
             "Adding ground coffee", 
             "ground coffee added",
             "Closing the lid",
             "lid closed"]
frame_count = 0
action_num = 0
current_text = ""  # Variable to store the current text to display
end_time = 0
act_seq=[]
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = round(frame_count / fps, 4)

    # Check if we're in the time range of the current action
    if action_num < len(start_times):
        if current_time >= start_times[action_num] and current_time <= end_times[action_num]:
            if current_time == start_times[action_num]:
                current_text = f"{pre_actions[action_num*2]}"
                if len(act_seq)==0 or act_seq[-1]!=current_text:
                    act_seq.append(current_text)
            if current_time == end_times[action_num]:
                current_text = f"{pre_actions[action_num*2+1]}"
                if act_seq[-1]!=current_text:
                    act_seq.append(current_text)
                action_num += 1  # Move to next action

    elif current_time>stime[0] and current_time<stime[-1]:
        current_text = "Press the start button"
        if act_seq[-1]!=current_text:
            act_seq.append(current_text)
        end_time = 1+current_time
    elif current_time>=stime[-1]:
        current_text = "Fnished!"
        if act_seq[-1]!=current_text:
            act_seq.append(current_text)



    # Display the current text
    if current_text:
        cv2.putText(frame, current_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)  # Save frame to output video
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(act_seq)