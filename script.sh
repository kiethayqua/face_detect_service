#!/bin/bash

# stop all containers
docker stop $(docker ps -a -q)

# remove all existed containers
docker rm $(docker ps -a -q)

# remove all images
docker rmi $(docker images -a -q)

# build docker image
docker build -t backend-facedetect-app .

# run docker image and expose server
docker run -p 80:80 backend-facedetect-app & ssh -o ServerAliveInterval=60 -R appx:80:0.0.0.0:80 serveo.net
