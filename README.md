# new_reid
#cuda9.0 python3.5
#pytorch
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ADD sources.list /etc/apt/

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget \
        curl \
        git \
        libopencv-dev\
        openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    /bin/bash Miniconda3-4.5.4-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-4.5.4-Linux-x86_64.sh 

ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardX

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.1.0

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchvision==0.3.0

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple dominate

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy==1.1.0

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python==4.1.0.25

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn==0.18.1

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple visdom==0.1.8.9 

WORKDIR /root
