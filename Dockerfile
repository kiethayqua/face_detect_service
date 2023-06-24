FROM animcogn/face_recognition:cpu
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt
RUN pip3 install "uvicorn[standard]"
COPY ./app /code/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
