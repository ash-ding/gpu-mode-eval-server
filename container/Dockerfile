FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu129

RUN pip3 install --no-cache-dir \
    triton==3.3.1 \
    numpy \
    pyyaml

WORKDIR /workspace

COPY pool_worker.py /workspace/pool_worker.py

ENTRYPOINT ["python3", "/workspace/pool_worker.py"]
