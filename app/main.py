from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import face_recognition
import shutil
import pathlib
import os
import cv2
import numpy as np
import uuid

app = FastAPI()
app.mount('/app/static', StaticFiles(directory="app/static"), name="static")


class People(object):
    name = ""
    enc = []

    def __init__(self, name, enc):
        self.name = name
        self.enc = enc


class DetectedResponse(object):
    phones = []
    img = ""

    def __init__(self, phones, img):
        self.phones = phones
        self.img = img


know_faces = []
know_face_encs = []


def init():
    for child in pathlib.Path('./app/data').iterdir():
        face_image = face_recognition.load_image_file('./' + str(child))
        face_name = os.path.splitext(str(child))[0].split('/')[-1]
        face_image_enc = face_recognition.face_encodings(face_image)[0]
        know_faces.append(People(face_name, face_image_enc))

    def get_encs():
        res = []
        for i in range(len(know_faces)):
            res.append(know_faces[i].enc)
        return res

    global know_face_encs
    know_face_encs = get_encs()


init()


def detect_face(unknown_face_file_path: str):
    phones = []

    origin_picture = cv2.imread(unknown_face_file_path)
    face_picture = face_recognition.load_image_file(unknown_face_file_path)
    face_locations = face_recognition.face_locations(
        face_picture)
    face_encodings = face_recognition.face_encodings(
        face_picture, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(know_face_encs, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(
            know_face_encs, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = know_faces[best_match_index].name
            phones.append(name)

        cv2.rectangle(origin_picture, (left, top),
                      (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(origin_picture, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = (right - left) / 200
        scaled_font_size = scale * 1.0
        cv2.putText(origin_picture, name, (left, bottom - 6),
                    font, scaled_font_size, (255, 255, 255), 1)

    random_uuid = uuid.uuid4()
    cv2.imwrite(f"./app/static/{random_uuid}.png", origin_picture)

    return DetectedResponse(phones, f"https://appx.serveo.net/app/static/{random_uuid}.png")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
        results = detect_face(file.filename)

    return {"data": results}


@app.get("/")
def read_root():
    return {"Hello": "World"}
