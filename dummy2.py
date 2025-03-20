import json
from collections import defaultdict

# Load the JSON data from the file
with open('movement_events.json', 'r') as file:
    data = json.load(file)

def generate_object_action(data):

    # Create a list to store the combined events
    moving_seq = {}
    prev = 99
    start_time = 0
    ori_pos = 0
    fin_pos = 0
    # state = 0 no object in motion, state = 1 object in motion, state = 2 object finished motion
    state = 0
    combined_events=[]
    for event in data:
        obj_id = event['object_id']
        from_position = event['from_position']
        to_position = event['to_position']
        last_time = event['last_time']
        # print(str(last_time-event['current_time']))
        if prev!=obj_id:
            if state != 1:
                start_time = last_time
                prev = obj_id
                ori_pos = from_position
                state = 1
                # print(obj_id)
                print('yes')
                print(obj_id)
            elif state ==1:
                # start_time = last_time
                print(obj_id)
                
                # Combine the data
                combined_event = {
                    'object_id': prev,
                    'start_time': start_time,
                    'finish_time': fin_pos,
                    'from_position': ori_pos,
                    'to_position': event['to_position'],
                }
                # 
                # prev = obj_id
                start_time = last_time
                prev = obj_id
                ori_pos = from_position
                # Add the combined event to the result list
                combined_events.append(combined_event)
                start_time = last_time
        fin_pos = event['current_time']
    return combined_events

# After processing the video, save the movement events to a JSON file
with open('combinedevents.json', 'w') as json_file:
    json.dump(combined_events, json_file, indent=4)