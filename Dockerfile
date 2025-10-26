FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf \
    TRANSFORMERS_CACHE=/workspace/hf

WORKDIR /app

RUN apt-get update && apt-get install -y git libgl1 poppler-utils && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic==2.* \
    pillow opencv-python pdf2image \
    transformers==4.44.2 accelerate==0.34.2 safetensors \
    torchvision bitsandbytes==0.43.1

# Optional: install xformers for faster attention if compatible with the GPU
# RUN pip install --no-cache-dir xformers==0.0.27.post2

COPY pod_server.py .

EXPOSE 8000
CMD ["uvicorn", "pod_server:app", "--host", "0.0.0.0", "--port", "8000"]
