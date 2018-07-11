FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

RUN apt-get -y update && \
    apt-get -y install \
        cmake libboost-all-dev wget unzip build-essential cmake git pkg-config libatlas-base-dev gfortran libgtk2.0-dev \
        libavcodec-dev libavformat-dev libswscale-dev libjpeg-dev libpng-dev libtiff-dev libv4l-dev python3 python3-pip

RUN pip3 install tensorflow-gpu==1.4.1 && \
    pip3 install git+https://github.com/Pithikos/python-websocket-server git+https://github.com/dikun-mv/tf-pose-estimation

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.1.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.1.zip && \
    unzip opencv.zip && rm opencv.zip && \
    unzip opencv_contrib.zip && rm opencv_contrib.zip && \
    mkdir opencv-3.4.1/build && cd opencv-3.4.1/build && \
    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
	    -D INSTALL_C_EXAMPLES=OFF \
	    -D BUILD_EXAMPLES=OFF \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-3.4.1/modules \
        -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
        -D WITH_CUDA=OFF \
        -D WITH_IPP=OFF \
        -D WITH_FFMPEG=ON \
        -D WITH_V4L=ON .. && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf opencv-3.4.1 && rm -rf opencv_contrib-3.4.1

ADD . /usr/src/app

CMD python3 /usr/src/app/src/main.py
