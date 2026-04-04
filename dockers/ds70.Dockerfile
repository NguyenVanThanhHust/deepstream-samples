FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch
ARG DEBIAN_FRONTEND=noninteractive

# 1. Broad cleanup of the problematic repo references
RUN grep -l "librealsense" /etc/apt/sources.list.d/* | xargs rm -f || true

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    checkinstall \
    locales \
    lsb-release \
    mesa-utils \
    subversion \
    vim \
    terminator \
    xterm \
    wget \
    htop \
    libssl-dev \
    build-essential \
    dbus-x11 \
    software-properties-common \
    gdb valgrind \
    libeigen3-dev \
    libboost-all-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt install -y python3-gi python3-dev python3-gst-1.0 python-gi-dev meson \
    python3-pip python3.10-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev \
    libcairo2-dev libgstreamer-plugins-base1.0-dev fish

# additional libs for deepstream
WORKDIR /opt/nvidia/deepstream/deepstream/
RUN ./user_additional_install.sh
RUN ./update_rtpmanager.sh
RUN python3 -m pip install opencv-python loguru confluent_kafka requests
RUN python3 -m pip install --upgrade google-api-python-client cuda-python build Pillow
RUN python3 -m pip install --force-reinstall protobuf==3.20.* numpy==1.26.0

# build opencv c++
WORKDIR /opt/
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
RUN unzip opencv.zip
WORKDIR /opt/opencv-4.8.0/build/
RUN cmake ..
RUN make -j8 && make install && ldconfig 
RUN rm /opt/opencv.zip

# deepstream python
WORKDIR /opt/nvidia/deepstream/deepstream/sources/
RUN git clone -b v1.1.11 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git --recursive --shallow-submodules
WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
RUN git submodule update --init

RUN cd 3rdparty/gstreamer/subprojects/gst-python/ && \
    meson setup build && \
    cd build && \
    ninja && \
    ninja install

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/build
RUN cmake .. && make -j10
RUN pip3 install ./pyds-*.whl

## Add ReID model
RUN mkdir /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
RUN wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt -P /opt/nvidia/deepstream/deepstream/samples/models/Tracker/

WORKDIR /opt/
RUN git clone https://github.com/p-ranav/argparse
WORKDIR /opt/argparse/build 
RUN cmake ..
RUN make -j4 && make install

RUN echo 'alias trtexec=/usr/src/tensorrt/bin/trtexec' >> ~/.bashrc
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo "alias ..='cd ..'" >> ~/.bashrc
RUN echo "alias ...='cd .. && cd ..'" >> ~/.bashrc
RUN echo "alias python=/usr/bin/python3" >> ~/.bashrc
RUN echo "alias p=/usr/bin/python3" >> ~/.bashrc

WORKDIR /workspace/