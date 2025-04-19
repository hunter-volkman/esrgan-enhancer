# Dockerfile for ESRGAN + Analytics Pipeline
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/Real-ESRGAN:/workspace/Real-ESRGAN/BasicSR:$PYTHONPATH

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        unzip \
        ffmpeg \
        libgl1-mesa-glx \
        libheif-examples \
        python3.10 \
        python3-pip \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /workspace

# Clone Real-ESRGAN and install dependencies in one layer
RUN git clone https://github.com/xinntao/Real-ESRGAN.git && \
    cd Real-ESRGAN && \
    echo "__version__ = '0.3.0'" > realesrgan/version.py && \
    # Install dependencies for Real-ESRGAN and our pipeline
    pip install --no-cache-dir \
        -r requirements.txt \
        basicsr \
        facexlib \
        gfpgan \
        matplotlib \
        scikit-image \
        opencv-python \
        tqdm \
        streamlit \
        lpips \
        pillow \
        tb-nightly && \
    # Install Real-ESRGAN
    python setup.py develop && \
    # Fix the DCNv2Pack issue
    sed -i '/class DCNv2Pack/,/^$/d' BasicSR/basicsr/archs/arch_util.py && \
    # Fix import issue in degradations.py if needed
    find . -name "degradations.py" -exec sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' {} \;

# Create directory structure
RUN mkdir -p /workspace/datasets/custom/train/hr \
             /workspace/datasets/custom/val/hr \
             /workspace/datasets/custom/val/lr \
             /workspace/results \
             /workspace/test_images

# Copy scripts
COPY scripts/ /workspace/scripts/
RUN chmod +x /workspace/scripts/*.sh

# Expose streamlit port
EXPOSE 8501

# Default working directory
WORKDIR /workspace

# Make scripts directory available in PATH
ENV PATH="/workspace/scripts:${PATH}"

# Environment verification command
RUN python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__); import realesrgan; print('Real-ESRGAN version:', realesrgan.__version__)"

# Default command
CMD ["/bin/bash"]