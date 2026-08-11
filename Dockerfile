# Base image with CUDA support (adjust CUDA version if needed)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /opt

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY src/ ./src/
COPY nnunet/ ./nnunet/
COPY run_model.py .

# (Optional) Set environment variables that nnU-Net often needs
ENV nnUNet_raw="/opt/nnunet/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/nnunet/nnUNet_preprocessed"
ENV nnUNet_results="/opt/nnunet/nnUNet_results"

# Default command (you can change this later if needed)
ENTRYPOINT ["python", "run_model.py"]