FROM python:3.8



# 设置当前目录为工作目录
WORKDIR /workspace


RUN apt update -q \
 && apt install -y -qq tzdata bash build-essential git curl wget software-properties-common \
    vim ca-certificates libffi-dev libssl-dev libsndfile1 libbz2-dev liblzma-dev locales \
    libboost-all-dev libboost-tools-dev libboost-thread-dev cmake \
    python3 python3-setuptools python3-pip cython


RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

