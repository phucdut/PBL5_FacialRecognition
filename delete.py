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



response = delete_face("minh222")
print(response.json())