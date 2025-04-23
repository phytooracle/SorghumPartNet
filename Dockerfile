# Use the official CUDA image as the base image
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip

# Install PyTorch 1.10.1 with CUDA 11.3
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install additional packages from seg.yml
COPY seg.yml /tmp/seg.yml
RUN pip3 install -r <(grep -E '^[^#]' /tmp/seg.yml | grep -E '^[^ ]+==[^ ]+$')

# Set the working directory
WORKDIR /opt

# Copy the project files into the container
COPY . /opt

# Set the entrypoint
ENTRYPOINT [ "/usr/local/bin/python3.8", "/opt/hpc.py" ]