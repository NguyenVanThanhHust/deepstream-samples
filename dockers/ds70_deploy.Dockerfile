FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN grep -l "librealsense" /etc/apt/sources.list.d/* | xargs rm -f || true

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    build-essential \
    cmake \
    meson \
    ninja-build \
    python3-dev \
    python3-pip \
    python3-gi \
    python3-gst-1.0 \
    python-gi-dev \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    autoconf automake libtool m4 \
    libeigen3-dev \
    libboost-all-dev \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1

RUN python3 -m pip install \
    numpy \
    protobuf \
    Pillow \
    requests \
    loguru \
    cuda-python \
    opencv-python \
    confluent_kafka \
    google-api-python-client \
    build

WORKDIR /opt/nvidia/deepstream/deepstream

RUN ./user_additional_install.sh
RUN ./update_rtpmanager.sh


WORKDIR /opt/nvidia/deepstream/deepstream/sources

RUN git clone \
    --depth 1 \
    --branch v1.1.11 \
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps

RUN git submodule update --init

RUN cd 3rdparty/gstreamer/subprojects/gst-python && \
    meson setup build && \
    cd build && \
    ninja && \
    ninja install

WORKDIR bindings/build

RUN cmake .. && \
    make -j$(nproc)

RUN pip install pyds-*.whl

WORKDIR /opt

RUN git clone --depth 1 https://github.com/p-ranav/argparse

WORKDIR /opt/argparse/build

RUN cmake .. && \
    make -j$(nproc) && \
    make install

FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-gi \
    python3-gst-1.0 \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
&& rm -rf /var/lib/apt/lists/*


COPY --from=builder \
/usr/local/lib/python3.10/dist-packages \
/usr/local/lib/python3.10/dist-packages

COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/include /usr/local/include
COPY --from=builder /usr/local/bin /usr/local/bin

RUN mkdir -p \
/opt/nvidia/deepstream/deepstream/samples/models/Tracker

RUN wget \
https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt \
-P /opt/nvidia/deepstream/deepstream/samples/models/Tracker/

RUN pip install \
    --no-cache-dir \
    numpy \
    protobuf \
    Pillow \
    requests \
    loguru \
    opencv-python \
    cuda-python \
    confluent_kafka \
    google-api-python-client \
    cupy-cuda12x