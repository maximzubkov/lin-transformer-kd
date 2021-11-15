FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y \
                       build-essential \
                       ca-certificates \
                       wget \
                       unzip \
                       ssh \
                       cmake \
                       git \
                       vim \
                       python3-dev python3-pip python3-setuptools

RUN ln -sf $(which python3) /usr/bin/python \
    && ln -sf $(which pip3) /usr/bin/pip

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /lin-transformer-kd
COPY . /lin-transformer-kd

RUN python -m pip install --upgrade pip
RUN pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
RUN pip install -r requirements.txt