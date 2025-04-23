
FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.8.20
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y

RUN apt-get update 
RUN apt-get install -y wget \
                       gdal-bin \
                       libgdal-dev \
                       libspatialindex-dev \
                       build-essential \
                       software-properties-common \
                       apt-utils \
                       libgl1-mesa-glx \
                       ffmpeg \
                       libsm6 \
                       libxext6 \
                       libffi-dev \
                       libbz2-dev \
                       zlib1g-dev \
                       libreadline-gplv2-dev \
                       libncursesw5-dev \
                       libssl-dev \
                       libsqlite3-dev \
                       tk-dev \
                       libgdbm-dev \
                       libc6-dev \
                       liblzma-dev \
                       libsm6 \
                       libxext6 \
                       libxrender-dev \
                       libgl1-mesa-dev   

# Download and install the CUDA Toolkit in /opt
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run -P /opt && \
    chmod +x /opt/cuda_11.3.1_465.19.01_linux.run && \
    /opt/cuda_11.3.1_465.19.01_linux.run --silent --toolkit --override && \
    rm /opt/cuda_11.3.1_465.19.01_linux.run

# Set environment variables for CUDA
ENV PATH=/opt/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

# Download and extract Python sources
RUN cd /opt \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \                                              
    && tar xzf Python-${PYTHON_VERSION}.tgz

# Build Python and remove left-over sources
RUN cd /opt/Python-${PYTHON_VERSION} \ 
    && ./configure --with-ensurepip=install \
    && make install \
    && rm /opt/Python-${PYTHON_VERSION}.tgz /opt/Python-${PYTHON_VERSION} -rf

RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 --index-url https://download.pytorch.org/whl/cu113
RUN pip3 install -r /opt/requirements.txt --no-cache-dir
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages:$PYTHONPATH


ENTRYPOINT [ "/usr/local/bin/python3.8", "/opt/hpc.py" ]