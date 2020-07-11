#!/bin/sh

# make sure we run from the root of the repository
[ "$(basename "$PWD")" == "cuda" ] && cd ../..
[ "$(basename "$PWD")" == "docker" ] && cd ..

IMAGE_NAME="${IMAGE_NAME:-vanitysearch}"
# default arguments
CCAP="${CCAP:-5.2}"
CUDA="${CUDA:-10.2}"

CAPP_MAJOR="${CCAP%.*}"

if [ "${CAPP_MAJOR}" -lt 5 ]; then
    # For 2.x and 3.x branches
    DOCKERFILE=./docker/cuda/ccap-2.0.Dockerfile
else
    DOCKERFILE=./docker/cuda/Dockerfile
fi

docker build \
    --build-arg "CCAP=${CCAP}" \
    --build-arg "CUDA=${CUDA}" \
    -t "${IMAGE_NAME}:cuda-ccap-${CCAP}" \
    -f "${DOCKERFILE}" .
