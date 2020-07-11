#!/bin/bash

export IMAGE_NAME="${IMAGE_NAME:-vanitysearch}"

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin

docker push "${IMAGE_NAME}":latest
docker push "${IMAGE_NAME}":cuda-ccap-6
docker push "${IMAGE_NAME}":cuda-ccap-6.0

docker push "${IMAGE_NAME}":cuda-ccap-5
docker push "${IMAGE_NAME}":cuda-ccap-5.2

docker push "${IMAGE_NAME}":cuda-ccap-2
docker push "${IMAGE_NAME}":cuda-ccap-2.0

docker push "${IMAGE_NAME}":cpu
