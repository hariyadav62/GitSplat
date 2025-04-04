# Use the official PyTorch image with CUDA 11.8 support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl wget gnupg build-essential \
    ninja-build git tzdata lsb-release ca-certificates gnupg python3-pip \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils python3.11-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Install latest pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Upgrade essential Python packaging tools
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# Install CUDA 11.8 Toolkit manually for nvcc and compiler support
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-8 && \
    ln -s /usr/local/cuda-11.8 /usr/local/cuda && \
    rm -f cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /InstantSplat

# Define CUDA architectures explicitly
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Copy dependencies first
COPY requirements.txt .

# Install Python dependencies
RUN python3.11 -m pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Install PyTorch compiled for CUDA 11.8
RUN python3.11 -m pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Copy full source code
COPY . .

# Ensure setuptools + packaging are wired correctly for cpp_extension
RUN python3.11 -m pip install --force-reinstall --no-cache-dir setuptools packaging

# Build and install CUDA-based submodules
RUN python3.11 -m pip install ./submodules/simple-knn \
    && python3.11 -m pip install ./submodules/diff-gaussian-rasterization \
    && python3.11 -m pip install ./submodules/fused-ssim

# Expose necessary port
EXPOSE 2052

# Start the app
CMD ["python3.11", "/InstantSplat/instantsplat_api.py"]
