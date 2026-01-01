FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch
ARG DEBIAN_FRONTEND=noninteractive

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
    python3-pip python3.12-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev \ 
    libcairo2-dev libgstreamer-plugins-base1.0-dev 

# additional libs for deepstream
WORKDIR /opt/nvidia/deepstream/deepstream/
RUN ./user_additional_install.sh
RUN ./update_rtpmanager.sh
RUN python3 -m pip install opencv-python loguru confluent_kafka requests
RUN python3 -m pip install --upgrade google-api-python-client cuda-python build Pillow
RUN python3 -m pip install --force-reinstall protobuf==3.20.*
RUN python3 -m pip install --force-reinstall numpy==1.26.0
# build opencv c++
WORKDIR /opt/
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
RUN unzip opencv.zip
WORKDIR /opt/opencv-4.8.0/build/
RUN cmake ..
RUN make -j12 && make install && ldconfig 

RUN rm /opt/opencv.zip

# deepstream python
WORKDIR /opt/nvidia/deepstream/deepstream/sources/
RUN git clone -b v1.2.2 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git --recursive --shallow-submodules
WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
RUN git submodule update --init
RUN python3 bindings/3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse

RUN cd bindings/3rdparty/gstreamer/subprojects/gst-python/ && \
    meson setup build && \
    cd build && \
    ninja && \
    ninja install

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings
RUN python3 -m build
RUN cd dist && pip3 install ./pyds-*.whl

RUN echo 'alias trtexec=/usr/src/tensorrt/bin/trtexec' >> ~/.bashrc
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo "alias ..='cd ..'" >> ~/.bashrc
RUN echo "alias ...='cd .. && cd ..'" >> ~/.bashrc
RUN echo "alias python=/usr/bin/python3" >> ~/.bashrc
RUN echo "alias p=/usr/bin/python3" >> ~/.bashrc

WORKDIR /workspace/

ARG USERNAME=thanhnv
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME