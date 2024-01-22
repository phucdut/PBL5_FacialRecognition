import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Initialize OpenCV video capture object for the default camera
cap = cv2.VideoCapture(0)

# Set API host and request type
api_host = 'http://0.0.0.0:8001/'
type_rq_image = 'img_object_detection_to_img'
type_rq_json = 'img_object_detection_to_json'
type_rq_learn = 'learn_face'
type_rq_save = 'save_face'
type_rq_delete = 'delete_face'

# Set video capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def set_schedule(data):
    # use post send data to server
    response = requests.post(api_host + "schedules", json=data)
    print(response.json())

def get_schedule():
    # use get get data from server
    response = requests.get(api_host + "schedules")
    _json_data = response.json()
    print(_json_data["title"])
    print(_json_data["name"])

def save_face(name):
    response = requests.post(api_host + type_rq_save, data={"name": name})
    return response

def delete_face(name):
    response = requests.post(api_host + type_rq_delete, data={"name": name})
    return response

def learn_face(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(api_host + type_rq_learn, files=files)
    return response

def recognize_face_to_image(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(api_host + type_rq_image, files=files)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)[:, :, ::-1].copy()
    return img

def recognize_face_to_json(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(api_host + type_rq_json, files=files)
    return response

cmd = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture video")
        break
    if cmd == 0:
        img = recognize_face_to_image(frame)
        cv2.imshow('detect to image', img)
    elif cmd == 1:
        txt = recognize_face_to_json(frame)
        print(txt)
        cv2.imshow('detect to json', frame)
    elif cmd == 2:
        response = learn_face(frame)
        print(response.json())
        cv2.imshow('learning', frame)
    elif cmd == 3:
        response = save_face("minh222")
        print(response.json())
        cmd = 0
    elif cmd == 4:
        response = delete_face("minh222")
        print(response.json())
        cmd = 0
    elif cmd == 5:
        get_schedule()
        cv2.imshow('Object Detection', frame)
    else:
        set_schedule({"title": "title for schedule", "name": "hello world"})
        cv2.imshow('Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('1'):
        cmd = 1
    if cv2.waitKey(1) & 0xFF == ord('2'):
        cmd = 2
    if cv2.waitKey(1) & 0xFF == ord('3'):
        cmd = 3
    if cv2.waitKey(1) & 0xFF == ord('4'):
        cmd = 4
    if cv2.waitKey(1) & 0xFF == ord('5'):
        cmd = 5
    if cv2.waitKey(1) & 0xFF == ord('0'):
        cmd = 0

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()