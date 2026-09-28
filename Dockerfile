FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        ca-certificates \
        build-essential \
        ninja-build \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dcsharp

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install \
        plyfile \
        tqdm \
        opencv-python-headless \
        imageio \
        matplotlib \
        joblib \
        scipy \
        lpips \
        tensorboard

# kiui is imported by train.py for orbit-camera videos. Install it without
# letting it replace the CUDA torch wheel above.
RUN python -m pip install kiui --no-deps \
    && python -m pip install \
        "numpy==1.26.4" \
        "plyfile==1.0.3" \
        "opencv-python-headless==4.10.0.84" \
        pillow \
        trimesh \
        lazy_loader \
        varname \
        objprint \
        rich \
        questionary \
        pyyaml

COPY submodules/simple-knn-v1 /opt/dcsharp/submodules/simple-knn-v1
COPY submodules/diff-gaussian-rasterization-new /opt/dcsharp/submodules/diff-gaussian-rasterization-new

RUN git clone --depth 1 --branch 0.9.9.8 https://github.com/g-truc/glm.git \
        /opt/dcsharp/submodules/diff-gaussian-rasterization-new/third_party/glm \
    && python -m pip uninstall -y diff_gaussian_rasterization simple_knn || true \
    && python -m pip install --no-build-isolation --force-reinstall --no-cache-dir /opt/dcsharp/submodules/simple-knn-v1 \
    && python -m pip install --no-build-isolation --force-reinstall --no-cache-dir /opt/dcsharp/submodules/diff-gaussian-rasterization-new \
    && python -c "from diff_gaussian_rasterization import _C; assert _C.sh_variant() == 'dcsh', _C.sh_variant()"

COPY . /opt/dcsharp

RUN chmod +x /opt/dcsharp/docker/entrypoint.sh /opt/dcsharp/train.sh /opt/dcsharp/rendering.sh /opt/dcsharp/full_eval.sh \
    && python /opt/dcsharp/docker/patch_kiui.py \
    && python -c "from torchvision.models import AlexNet_Weights, alexnet; alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)"

WORKDIR /opt/dcsharp
ENTRYPOINT ["/opt/dcsharp/docker/entrypoint.sh"]
CMD ["help"]
