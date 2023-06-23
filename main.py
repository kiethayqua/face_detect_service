from typing import Union
from fastapi import FastAPI, File, UploadFile
import face_recognition
import shutil
import pathlib
import os

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

    unknown_picture = face_recognition.load_image_file(unknown_face_file_path)
    unknown_picture_locations = face_recognition.face_locations(
        unknown_picture)
    unknown_face_encs = face_recognition.face_encodings(
        unknown_picture, unknown_picture_locations)

    detected_indexes = []
    for i in range(len(unknown_face_encs)):
        matches = face_recognition.compare_faces(
            know_face_encs, unknown_face_encs[i])
        if True in matches:
            first_index = matches.index(True)
            if first_index not in detected_indexes:
                results.append(know_faces[first_index].name)
                detected_indexes.append(first_index)

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


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
