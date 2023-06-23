from fastapi import FastAPI, File, UploadFile
import face_recognition
import shutil
import pathlib
import os
import uvicorn
import cv2
import numpy as np

app = FastAPI()


class People(object):
    name = ""
    enc = []

    def __init__(self, name, enc):
        self.name = name
        self.enc = enc


know_faces = []
know_face_encs = []


def init():
    for child in pathlib.Path('./data').iterdir():
        face_image = face_recognition.load_image_file('./' + str(child))
        face_name = os.path.splitext(str(child))[0]
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
    results = []

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

        cv2.rectangle(origin_picture, (left, top),
                      (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(origin_picture, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(origin_picture, name, (left + 6, bottom - 6),
                    font, 2.0, (255, 255, 255), 2)

    cv2.imwrite("final.png", origin_picture)

    return results


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
        results = detect_face(file.filename)

    return {"people": results}


@app.get("/")
def read_root():
    return {"Hello": "World"}


# this code support serveo.net server
# ssh -R 80:localhost:3000 serveo.net
if __name__ == '__main__':
    uvicorn.run(app, port=3000, host='0.0.0.0')
