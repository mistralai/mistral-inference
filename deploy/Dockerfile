FROM --platform=amd64 nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04 as base

WORKDIR /workspace

RUN apt update && \
    apt install -y python3-pip python3-packaging \
    git ninja-build && \
    pip3 install -U pip

# Tweak this list to reduce build time
# https://developer.nvidia.com/cuda-gpus
ENV TORCH_CUDA_ARCH_LIST "7.0;7.2;7.5;8.0;8.6;8.9;9.0"

RUN pip3 install "torch==2.1.1"

# This build is slow but NVIDIA does not provide binaries. Increase MAX_JOBS as needed.
RUN pip3 install "git+https://github.com/stanford-futuredata/megablocks.git"
RUN pip3 install "git+https://github.com/vllm-project/vllm.git"
RUN pip3 install "xformers==0.0.23" "transformers==4.36.0" "fschat[model_worker]==0.2.34"

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82 && \
    sed -i '/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/d' setup.py && \
    python3 setup.py install --cpp_ext --cuda_ext


COPY entrypoint.sh .

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]