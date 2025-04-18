# Dockerfile for ESRGAN + Analytics Pipeline
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        unzip \
        ffmpeg \
        libgl1 \
        python3.10 \
        python3-pip \
        python3.10-venv && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace


# Clone Real-ESRGAN
# RUN git clone https://github.com/xinntao/Real-ESRGAN.git && \
#    cd Real-ESRGAN && \
#    pip install -r requirements.txt && \
#    python setup.py develop

# Install additional packages
RUN pip install \
    matplotlib \
    scikit-image \
    opencv-python \
    tqdm \
    streamlit \
    lpips \
    pillow \
    tb-nightly

# Copy scripts
COPY scripts/ /workspace/scripts/
RUN chmod +x /workspace/scripts/*.sh

# Expose streamlit port
EXPOSE 8501

# Default command
CMD ["/bin/bash"]
