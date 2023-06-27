FROM animcogn/face_recognition:cpu
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt
RUN pip3 install "uvicorn[standard]"
RUN pip3 install wget
COPY ./app /code/app

# download mock data
RUN python3 -m wget https://i.ndh.vn/2021/12/21/a4-1640071043.jpg -o /code/app/data/0918189999.jpg
RUN python3 -m wget https://i.ndh.vn/2021/12/21/a5-1640071216.jpg -o /code/app/data/0976375050.jpg
RUN python3 -m wget https://i.ndh.vn/2021/12/21/a6-1640071730.jpg -o /code/app/data/0908045788.jpg
RUN python3 -m wget https://i.ndh.vn/2021/12/21/cuong-do.jpg -o /code/app/data/0912482025.jpg
RUN python3 -m wget https://i.ndh.vn/2021/12/21/a9-1640073023.jpg -o /code/app/data/0983050910.jpg
RUN python3 -m wget https://i.ndh.vn/2021/12/21/a10-1640073655.jpg -o /code/app/data/0936246698.jpg


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]