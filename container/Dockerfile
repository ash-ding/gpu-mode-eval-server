FROM docker.io/nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir \
    torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129

RUN pip3 install --no-cache-dir \
    triton==3.6.0 \
    numpy \
    pyyaml

WORKDIR /workspace

COPY pool_worker.py /workspace/pool_worker.py

ENTRYPOINT ["python3", "/workspace/pool_worker.py"]
