# Use the official CUDA image as the base image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04


# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 1.10.1 with CUDA 11.3
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install additional packages from seg.yml
RUN pip3 install cloudpickle==2.2.0 deprecation==2.1.0 google-auth==1.35.0 hyperopt==0.2.7 ipympl==0.9.2 jupyter-packaging==0.12.0 \
                mpmath==1.2.1 networkx==2.8.8 ninja==1.10.2.3 py4j==0.10.9.7 pyquaternion==0.9.9 setuptools==65.3.0 sympy==1.9 \
                tensorboard==2.10.1 tensorboard-data-server==0.6.1 tomlkit==0.11.4 torch-summary==1.4.5 torch-tb-profiler==0.3.1 \
                torchcluster==0.1.4 pytorch-lightning==1.5.8 h5py==3.6.0 torchmetrics==0.6.2 plyfile==0.7.4

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Install additional dependencies
RUN pip3 install --force-reinstall protobuf==3.15.8 --retries 10
# RUN wget https://files.pythonhosted.org/packages/10/02/4f777e87131b3e111f91e16c1782da707b8b6a5777dbbe11a19e0dc7d1a3/open3d-0.15.2-cp38-cp38-manylinux_2_27_x86_64.whl
# RUN pip3 install open3d-0.15.2-cp38-cp38-manylinux_2_27_x86_64.whl
RUN pip3 install open3d==0.16.0
RUN pip3 install --force-reinstall setuptools==59.5.0 
RUN pip3 install --force-reinstall numpy==1.23.5
RUN export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Set the working directory
WORKDIR /opt

# Copy the project files into the container
COPY . /opt

# Set PYTHONPATH to include all relevant directories
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:$PYTHONPATH

# Set the entrypoint
ENTRYPOINT ["/usr/bin/python3.10", "/opt/hpc.py"]