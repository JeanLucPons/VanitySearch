#!/bin/sh

# make sure we run from the root of the repository
[ "$(basename "$PWD")" == "cpu" ] && cd ../..
[ "$(basename "$PWD")" == "docker" ] && cd ..

IMAGE_NAME="${IMAGE_NAME:-vanitysearch}"

docker build \
    -t "${IMAGE_NAME}:cpu" \
    -t "${IMAGE_NAME}:latest" \
    -f ./docker/cpu/Dockerfile .
