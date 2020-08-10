#!/bin/sh

# make sure we run from the root of the repository
[ "$(basename "$PWD")" == "docker" ] && cd ..

export IMAGE_NAME="${IMAGE_NAME:-vanitysearch}"

env CCAP=6.0 CUDA=10.2 ./docker/cuda/build.sh
docker tag "${IMAGE_NAME}":cuda-ccap-6.0 "${IMAGE_NAME}":latest
docker tag "${IMAGE_NAME}":cuda-ccap-6.0 "${IMAGE_NAME}":cuda-ccap-6

env CCAP=5.2 CUDA=10.2 ./docker/cuda/build.sh
docker tag "${IMAGE_NAME}":cuda-ccap-5.2 "${IMAGE_NAME}":cuda-ccap-5

env CCAP=2.0 CUDA=8.0 ./docker/cuda/build.sh
docker tag "${IMAGE_NAME}":cuda-ccap-2.0 "${IMAGE_NAME}":cuda-ccap-2

./docker/cpu/build.sh
