FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

RUN apt-get -y update && \
    apt-get -y install libglib2.0-0 git python3 python3-pip && \
    pip3 install --upgrade pip

RUN pip3 install tensorflow-gpu==1.4.1 keras==2.1.5 h5py opencv-python-headless scikit-learn && \
    pip3 install git+https://github.com/ildoonet/tf-pose-estimation.git

ADD . /usr/src/app

CMD python3 /usr/src/app/src/main_webcam.py