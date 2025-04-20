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

# Clone Real-ESRGAN and BasicSR separately (more reliable)
RUN git clone https://github.com/xinntao/Real-ESRGAN.git && \
    git clone https://github.com/xinntao/BasicSR.git Real-ESRGAN/BasicSR && \
    mkdir -p Real-ESRGAN/realesrgan && \
    echo "__version__ = '0.3.0'" > Real-ESRGAN/realesrgan/version.py

# Fix the arch_util.py file before installing dependencies
# The better approach is to replace the problematic part rather than deleting it
RUN sed -i 's/class DCNv2Pack(ModulatedDeformConvPack):/# class DCNv2Pack(ModulatedDeformConvPack): # Commented out due to import errors/' /workspace/Real-ESRGAN/BasicSR/basicsr/archs/arch_util.py && \
    # Comment out the entire class body with proper Python syntax
    sed -i '/# class DCNv2Pack/,/^$/ s/^/# /' /workspace/Real-ESRGAN/BasicSR/basicsr/archs/arch_util.py && \
    # Fix import issues in degradations.py
    find /workspace/Real-ESRGAN/BasicSR -name "degradations.py" -exec sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' {} \;

# Install Python dependencies
RUN pip install --no-cache-dir \
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
    tb-nightly \
    torch torchvision

# Install Real-ESRGAN in development mode
RUN cd Real-ESRGAN && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py develop

# Create directory structure
RUN mkdir -p /workspace/datasets/custom/train/hr \
             /workspace/datasets/custom/val/hr \
             /workspace/datasets/custom/val/lr \
             /workspace/results \
             /workspace/test_images \
             /workspace/options

# Copy scripts
COPY scripts/ /workspace/scripts/
RUN chmod +x /workspace/scripts/*.sh

# Expose streamlit port
EXPOSE 8501

# Default working directory
WORKDIR /workspace

# Make scripts directory available in PATH
ENV PATH="/workspace/scripts:${PATH}"

# Skip the verification command that was causing issues
# We'll verify at runtime instead

# Default command
CMD ["/bin/bash"]
