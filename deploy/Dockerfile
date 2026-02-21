FROM --platform=amd64 nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-packaging \
        python3-venv \
        python3-dev \
        git \
        ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir -U pip

# Tweak this list to reduce build time
# https://developer.nvidia.com/cuda-gpus
ENV TORCH_CUDA_ARCH_LIST="7.0;7.2;7.5;8.0;8.6;8.9;9.0"

# torch in its own layer for reusable cache
RUN pip install --no-cache-dir "torch==2.4.0"

RUN pip install --no-cache-dir "wheel" "numpy"

RUN pip install --no-cache-dir --no-build-isolation \
    "megablocks==0.6.1" \
    "vllm==0.5.5" \
    "transformers==4.44.0" \
    "fschat[model_worker]==0.2.36"

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82 && \
    sed -i '/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/d' setup.py && \
    python3 setup.py install --cpp_ext --cuda_ext


FROM --platform=amd64 nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY entrypoint.sh .

RUN chmod +x /workspace/entrypoint.sh && \
    useradd -m -u 1000 mistral && \
    chown -R mistral:mistral /workspace

USER mistral

ENTRYPOINT ["/workspace/entrypoint.sh"]
