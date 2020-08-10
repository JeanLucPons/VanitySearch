# Multistage docker build, requires docker 17.05

# CUDA SDK version, and also a prefix of images' tag.
# Check out the list of officially supported tags:
# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
# Format: x.y, e.g.: "10.2".
# Required argument.
ARG CUDA

# builder stage
FROM nvidia/cuda:${CUDA}-devel as builder

# Install newer version of g++ than what Ubuntu 16.04 provides.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
    && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        g++-7 \
    && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

# CUDA Computational Capability.
# Format: x.y, e.g.: "5.2".
# Required argument.
ARG CCAP

RUN cd /app && \
  make \
  CXX=/usr/bin/g++-7 \
  CUDA=/usr/local/cuda \
  CXXCUDA=/usr/bin/g++ \
  gpu=1 \
  "CCAP=${CCAP}" \
  all

# runtime stage
FROM nvidia/cuda:${CUDA}-runtime

COPY --from=builder /app/VanitySearch /usr/bin/VanitySearch

ENTRYPOINT ["/usr/bin/VanitySearch"]
