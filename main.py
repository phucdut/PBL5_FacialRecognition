####################################### IMPORT #################################
import json
import pandas as pd
from PIL import Image
from loguru import logger
import sys
import io

# import jsonfrom typing import Annotated
from pydantic import BaseModel
from fastapi import FastAPI, Form

from fastapi import FastAPI, File, status, Body
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
from io import BytesIO

import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import sys
import json
import numpy as np

# from app import get_image_from_bytes
# from app import get_bytes_from_image
FRGraph = FaceRecGraph()
MTCNNGraph = FaceRecGraph()
aligner = AlignCustom()
extract_feature = FaceFeature(FRGraph)
face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2)

# Nhận diện và trả về kết quả nhận diện - (mảng đặc trưng, vị trí tương ứng, ngưỡng, ngưỡng phần trăm)
def findPeople(features_arr, positions, thres = 0.6, percent_thres = 70):
    f = open('./dataset.db','r')
    data_set = json.loads(f.read())
    returnRes = []
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown"
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]]
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance
                    result = person
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
    return returnRes   

# Nhận diện trong 1 ảnh và trả về hình ảnh đã được vẽ khung nhận diện và gán nhãn kết quả nhận diện - (khung ảnh)
def recognize_in_img(frame):
    rects, landmarks = face_detect.detect_face(frame,80)
    aligns = []
    positions = []
    recog_data = []

    for (i, rect) in enumerate(rects):
        aligned_face, face_pos = aligner.align(160,frame,landmarks[:,i])
        if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
            aligns.append(aligned_face)
            positions.append(face_pos)  
        else: 
            print("Align face failed") #log        
    if(len(aligns) > 0):
        features_arr = extract_feature.get_features(aligns)
        recog_data = findPeople(features_arr,positions)
        for (i,rect) in enumerate(rects):
            # get info from json array in file name infodata.json
            id = recog_data[i][0]
            name = ""
            ages = 0
            with open('infodata.json') as json_file:
                data = json.load(json_file)
                for p in data:
                    if str(p['id']) == id:
                        name = p['name']
                        ages = p['ages']
                        break

            disp = name + " - "+str(ages) +" - "+str(recog_data[i][1])+"%"
            if len(name) < 1:
                disp = "Unknown"
            if (ages) < 1:
                disp = "Unknown"
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0)) #draw bounding box for the face
            cv2.putText(frame, disp ,(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

    return frame, recog_data

# Nhận diện trong 1 ảnh và trả về kết quả dưới dạng dữ liệu text - (khung ảnh)
def recognize_in_text(frame):
    rects, landmarks = face_detect.detect_face(frame,80)
    aligns = []
    positions = []
    recog_data = []

    for (i, rect) in enumerate(rects):
        aligned_face, face_pos = aligner.align(160,frame,landmarks[:,i])
        if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
            aligns.append(aligned_face)
            positions.append(face_pos)
        else: 
            print("Align face failed") #log        
    if(len(aligns) > 0):
        features_arr = extract_feature.get_features(aligns)
        recog_data = findPeople(features_arr,positions)
        for (i,rect) in enumerate(rects):
            cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0)) #draw bounding box for the face
            cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

    return recog_data

# Chuyển đổi dữ liệu hình ảnh dạng bytes thành đối tượng Image - (file)
def get_image_from_bytes(binary_image: bytes) -> Image:

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    # save the image in JPEG format with quality 85

    return input_image


# Chuyển đổi dữ liệu hình ảnh dạng đối tượng Image thành dạng bytes - (image)
def get_bytes_from_image(image: Image) -> bytes:

    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image


####################################### logger #################################
# Cấu hình logger trong ứng dụng
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title
#Tạo 1 đối tượng app - bao gồm Tiêu đề, Mô tả ngắn và version
app = FastAPI(
    title="Object Detection FastAPI Template",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2023.1.31",
)

# Định nghĩa danh sách các nguồn được truy cập
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

#Thêm 1 middleware vào ứng dụng - CORS Cross-Origin Resource Sharing 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# tạo file JSON chứa thông tin về giao diện API
@app.on_event("startup")
def save_openapi_json():
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)



# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


class Item(BaseModel):
    name: str
    id: int
    state: str
    date: str
    time: str

# redirect
# Nhận 1 đối tượng item rồi thêm vào tệp schedules.csv
@app.post("/schedules")
async def set_schedule(item: Item):
    json_compatible_item_data = jsonable_encoder(item)
    # check file is exist create if not exist and add header
    try:
        df = pd.read_csv('schedules.csv')
    except:
        df = pd.DataFrame(columns=['name', 'id', 'state', 'date', 'time'])
        df.to_csv('schedules.csv', mode='a', header=True, index=False)
    

    df = pd.DataFrame(json_compatible_item_data, index=[0])
    df.to_csv('schedules.csv', mode='a', header=False, index=False)

    

    return {"state": "done"}


# Đọc dữ liệu từ tệp rồi chuyển nó thành bảng
@app.get("/schedules")
async def set_schedule():
    try:
        df = pd.read_csv('schedules.csv')
        dic = df.to_dict('records')
        return dic
    except:
        return []
    # dic = {"title": "title for schedule", "name": "hello world"}
    # return dic

# Nhận 1 tệp hình ảnh, thực hiện nhận diện, trả về hình ảnh đã nhận diện dưới dạng StreamingResponse cùng với thông tin
@app.post("/img_object_detection_to_img")
async def img_object_detection_to_img(file: bytes = File(...)):
    file_bytes = np.asarray(bytearray(file), dtype=np.uint8)  # Chuyển dữ liệu bytes -> numpy(uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Giải mã hình ảnh bằng thư viện trả về kết quả là biến img
    image, info = recognize_in_img(img) # Nhận diện đối tượng -> hình ảnh nhận diện và thông tin
    respone_header = {}
    if (len(info)):
        id = info[0][0]
        percentage = info[0][1]
        name = ""
        ages = 0
        # get info from json array in file name infodata.json
        with open('infodata.json') as json_file:
            data = json.load(json_file)
            for p in data:
                if str(p['id']) == id:
                    name = p['name']
                    ages = p['ages']
                    respone_header = {"name": name, "ages": ages, "percentage": percentage, "id": id}
                    break 
    cv2.imwrite("test.jpg", image)
    image = Image.open("test.jpg")
    return StreamingResponse(content=get_bytes_from_image(image), media_type="image/jpeg", headers={"info": json.dumps(respone_header)}) # Trả về nội dung ảnh dưới dạng bytes và thông tin nhận diện

# Nhận 1 tệp hình ảnh, thực hiện nhận diện, trả về kết quả nhận diện dưới dạng JSON
@app.post("/img_object_detection_to_json")
async def img_object_detection_to_json(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)

    file_bytes = np.asarray(bytearray(file), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    txt = recognize_in_text(img)
    return txt

person_imgs = {"Left" : [], "Right": [], "Center": []}      # Lưu trữ hình ảnh trong 3 vị trí
person_features = {"Left" : [], "Right": [], "Center": []}  # Lưu trữ các đặt trưng ở 3 vị trí tương ứng

@app.post("/learn_face")
async def learn_face(file: bytes = File(...)):
    file_bytes = np.asarray(bytearray(file), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rects, landmarks = face_detect.detect_face(frame, 80)   #Sử dụng phát hiện khuông mặt đê tìm vùng có khuôn mặt 
    for (i, rect) in enumerate(rects):          # -> rects: danh sách các hình chữ nhật chứa khuôn mặt và landmarks: vị tí các điểm đặt trưng
        cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,255)) #vẽ hộp giới hạn cho khuôn mặt là màu vàng

        aligned_frame, pos = aligner.align(160,frame,landmarks[:,i])
        if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
            person_imgs[pos].append(aligned_frame)

    cv2.imwrite("test.jpg", frame)
    image = Image.open("test.jpg")
    return StreamingResponse(content=get_bytes_from_image(image), media_type="image/jpeg")


@app.post("/save_face")  
async def save_face(name: str = Form(...), id: int = Form(...), ages: int = Form(...)):     # Nhận vào name, id, ages
    global person_imgs, person_features

    if len(person_imgs["Left"]) == 0 or len(person_imgs["Right"]) == 0 or len(person_imgs["Center"]) == 0:
        return {"state": "empty"}  # Kiểm tra xem có đủ 3 vị trí khuôn mặt k, nếu thiếu 1 trong 3 trả về empty
    
    f = open('./dataset.db','r')
    data_set = json.loads(f.read())
    for pos in person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
                                        # Tính giá trị trung bình của các đặc trưng chuyển thành list rồi lưu vào person_features
    data_set[id] = person_features      # Thêm khóa id và person_features vào data_set
    f = open('./dataset.db', 'w')
    f.write(json.dumps(data_set))       
    person_imgs = {"Left" : [], "Right": [], "Center": []}      
    person_features = {"Left" : [], "Right": [], "Center": []}

    # save info to json array in file name infodata.json
    with open('infodata.json') as json_file:    # Ghi lại nôi dung vào file infodata.json
        data = json.load(json_file)
        temp = data
        id_exist = False
        for p in data:
            if p['id'] == id:
                id_exist = True
                break
        if not id_exist:
            temp.append({
                "id": id,
                "name": name,
                "ages": ages
            })
            with open('infodata.json', 'w') as outfile:
                json.dump(temp, outfile)
        else:
            return {"state": "exist"}
    return {"state": "done"}


@app.post("/delete_face")
async def delete_face(id: int = Form(...)):

    # delete info to json array in file name infodata.json
    with open('infodata.json') as json_file:
        data = json.load(json_file)
        temp = data
        for p in data:
            if p['id'] == id:
                temp.remove(p)
                with open('infodata.json', 'w') as outfile:
                    json.dump(temp, outfile)
                break
    f = open('./dataset.db','r')
    data_set = json.loads(f.read())
    try:
        del data_set[str(id)]
        f = open('./dataset.db', 'w')
        f.write(json.dumps(data_set))
        return {"state": "done"}
    except:
        return {"state": "failed"}

    