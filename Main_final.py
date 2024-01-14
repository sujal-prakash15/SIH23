import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_sort1.deep_sort import DeepSort
from deep_sort1.sort.tracker import Tracker
# from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from ultralytics import YOLO
import time
from twilio.rest import Client
import datetime 
import math
import winsound
#CWD = os.getcwd() 61 31 track
video_path = os.path.join('.', 'data', 'test69.mp4')
video_out_path = os.path.join('.', 'testing1.webm')
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'VP90'), cap.get(cv2.CAP_PROP_FPS),(frame.shape[1], frame.shape[0]))

#######################  YOLO MODEL
model = YOLO("yolov8n.pt")


#######################  FIRE MODEL
modelf = YOLO('weights/best.pt')


############################
deep_sort_weight='deep_sort1/deep/checkpoint/ckpt.t7'
tracker=DeepSort(model_path=deep_sort_weight,max_age=30)




####################### Track Trespass System Test3
height, width = 1280,720
center_x, center_y = width - (width // 4), height - (height // 3)
center_y +=40
angle_degrees = 125
angle_radians = np.deg2rad(angle_degrees)
line_length = max(width, height) // 2
line_length -=70
start_x = center_x
start_y = center_y
end_x = int(center_x + line_length * np.cos(angle_radians))
end_y = int(center_y - line_length * np.sin(angle_radians))  # Negative sign due to the inverted y-axis




####################### Track Trespass System Test6
# height, width = 1080,612
# center_x, center_y = 0, height
# center_y-=50
# angle_degrees = 51
# angle_radians = np.deg2rad(angle_degrees)
# # line_length = max(width, height)
# line_length =height - height//3
# start_x = center_x
# start_y = center_y
# end_x = int(center_x + line_length * np.cos(angle_radians))
# end_y = int(center_y - line_length * np.sin(angle_radians))  # Negative sign due to the inverted y-axis


line_color = (0, 0, 255)  # Red color






####################### # ALERT SYSTEM Twilio
account_sid = 'AC03913f847cf0a3e683ab12d1f15a8b5e'
auth_token = 'f48db6ae0d562a26b7d09280247f0613'
twilio_number = ''
cctv_id = 'CCTV-DEF-002'
sending_time = []
firstSent = True
# Twilio Client
client = Client(account_sid, auth_token)

# List of phone numbers to send alerts to
phone_numbers = ['+918920692261']
################# alarm Beep alert
def alarm():
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
##############



#######################
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
detection_threshold = 0.5
integer_keys = list(range(80))
values = [
   "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]





###################################
# Data Extraction Here
fsc_object = {'fs_confidence':[]}
fsc_frame = pd.DataFrame(fsc_object)
ts_object = {'timestamp':[]}
ts_frame = pd.DataFrame(ts_object)
confidence_list = []
person_list = []
timestamp_list = []
###################################


class_map = {key: value for key, value in zip(integer_keys, values)}            #maps integer to class labels

# --------------------
##########################                                                      #alarm generate
# Initialize pyttsx3 for voice alerts
# alarm_sound = pyttsx3.init('sapi5')
# voices = alarm_sound.getProperty('voices')
# alarm_sound.setProperty('voice', voices[0].id)
# alarm_sound.setProperty('rate', 150)
# crowd_alert_triggered = False
#
# def crowd_alert():
#     global crowd_alert_triggered
#     alarm_text = "Crowd detected!"
#     alarm_sound.say(alarm_text)
#     alarm_sound.runAndWait()
#     crowd_alert_triggered = True
##############################




def send_alert(message,number):
     client = Client(account_sid, auth_token)
     message = client.messages.create(
  from_='+18508212615',
  body=message,
  to=number
)

tracked_cnt=0 # tresspassing time
thresh_cnt=3 #tresspassing threshold
################################################################
while ret:
    #########################################
    resultsfire = modelf.predict(source=frame, conf=0.20)  # fire detection model
    bounding_boxes = resultsfire[0].boxes.xyxy  # Assuming xyxy format for bounding boxes
    confidences = resultsfire[0].boxes.conf
    class_labels = resultsfire[0].boxes.cls
    cv2.line(frame, (start_x, start_y), (end_x, end_y), line_color, thickness=2) ## draw line custom

    for box, confidence, class_label in zip(bounding_boxes, confidences, class_labels):
        x_min, y_min, x_max, y_max = box.tolist()
        confidence = confidence.item()
        class_label = int(class_label.item())
        # print(f"Fire&smoke confidence:{confidence}")
        new_row = {'fs_confidence':confidence}
        fsc_frame = pd.concat([fsc_frame,pd.DataFrame([new_row])], ignore_index=True)
        confidence_list.append(confidence)
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        if (class_label == 0):
            color = (255, 0, 255)
        else:  ##shayad 1 smoke ke liye hain
            color = (255, 0, 0)


        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)                    #to draw fire detections
   
   
   
   
    #######################
    results = model(frame)
    num_keys = 80                                                                           # number of class labels in coco
    dict = {key: 0 for key in range(num_keys)}                                             # this dict maps integer to detection class counts
    for result in results:                                                                  # results for the coco model
        bboxes_xywh=[]
        confidence=[]
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            w=x2-x1
            h=y2-y1
            bbox_xywh=[x1,y1,w,h]
            bboxes_xywh.append(bbox_xywh)
            confidence.append(score)

            class_id = int(class_id)
            dict[class_id] += 1                                                             # updates detection count of particular class

            # if score > detection_threshold:
            # detections.append([x1, y1, x2, y2, score])

        filtered_dict = {class_map[key]: value for key, value in dict.items() if value != 0}  # filtered dict will only contain those pairs !=0 count
        for key, value in filtered_dict.items():
            print(f"{key}: {value}")

  
        # Convert lists to numpy arrays

        tracks = tracker.update(bboxes_xywh, confidence, frame)

        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            bbox_xywh = np.array(track.to_tlwh())  # Convert to NumPy array

            x, y, w, h = bbox_xywh  # Extract x, y, width, and height

            shift_per = 0.5
            y_shift = int(h * shift_per)
            x_shift = int(w * shift_per)
            y += y_shift
            x += x_shift
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (colors[track_id % len(colors)]), 3)

            # center coordinates
            x_center = int(x + w / 2)
            y_center = int(y + h / 2)


            ##############################################
            # Track Tracker Twilio here notification
            if y_center-x*math.tan(angle_radians) >= start_y - start_x*math.tan(angle_radians) : # Tracker send
                alarm()
                if firstSent:
                    firstSent = False
                    message = f"Track Tresspasser Detected at {cctv_id}"
                    test_number = '+918920692261'
                    send_alert(message,test_number)
           
           

            # Check if the calculated center coordinates are within the frame bounds
            if 0 <= x_center < frame.shape[1] and 0 <= y_center < frame.shape[0]:
                center_color = frame[y_center, x_center]

                try:
                    # Ensure that center_color contains valid numeric values
                    center_color = tuple(map(int, center_color))  # Convert color to integer values
                    cv2.circle(frame, (x_center, y_center), 10, center_color, -1)
                except Exception as e:
                    print(f"Error processing track {track_id}: {e}")
            else:
                print(f"Center coordinates ({x_center}, {y_center}) are out of bounds.")
            #############################





            text_annotations = [(key, value) for key, value in
                                filtered_dict.items()]  # maps label to their corresponding detection counts
            cv2.rectangle(frame, (0, 0), (250, 180), (222, 49, 99), -1)
            y = 60
            for key, value in text_annotations:
                cv2.putText(frame, f"{key}: {value}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y += 30  # if new label detected then increment y
               
               
               
               
               
                ###################                                                     #alarm sound
                # if key == 'person':
                #     crowd_threshold = 30  # Set your desired crowd threshold
                #     if value >= crowd_threshold and not crowd_alert_triggered:
                #         crowd_alert_thread = threading.Thread(target=crowd_alert)
                #         crowd_alert_thread.start()
                ##################

            # # cv2.putText(frame,str(len(detections)),(25,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    # if tracked_cnt>=1:
    #     tracked_cnt=0
    #     alarm()
    cap_out.write(frame)
    ret, frame = cap.read()
    # if cv2.waitKey(3) & 0xFF==ord('t'):
    #     alarm^=True
    #     tracked_cnt=0
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
# data_out_frame = pd.concat([fsc_frame,ts_frame,person_frame],axis=1)
# data_out_frame.to_csv('test12.csv', index=False)

  # ----------------------
cap.release()
cap_out.release()
cv2.destroyAllWindows()