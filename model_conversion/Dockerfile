FROM python:3.10-slim

# System deps for OpenCV headless and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone rPPG-Toolbox (we need the model definitions)
RUN git clone --depth 1 https://github.com/ubicomplab/rPPG-Toolbox.git /workspace/rPPG-Toolbox

# Download UltraFace ONNX model (~1.2MB)
COPY download_ultraface.py .
RUN python download_ultraface.py

# Copy conversion scripts
COPY convert_efficientphys.py .
COPY convert_factorizephys.py .
COPY convert_mediapipe.py .

# Output directory (mount host's public/ here)
VOLUME /output

# Default: run all conversions and copy outputs
CMD python convert_efficientphys.py && \
    python convert_factorizephys.py && \
    python convert_mediapipe.py && \
    cp /workspace/efficientphys.onnx /output/ && \
    cp /workspace/factorizephys.onnx /output/ && \
    cp /workspace/ultraface_320.onnx /output/ && \
    cp /workspace/blazeface_128.onnx /output/ && \
    echo "✅ All models copied to /output/"
