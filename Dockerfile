# =============================================================================
#  CrossTooth (CVPR 2025) -- Runpod Serverless worker
#  Stage 1 compiles the pointops CUDA extension (needs nvcc -> "devel" image),
#  Stage 2 is the slimmer "runtime" image that actually serves requests.
# =============================================================================
ARG PYTORCH_TAG=2.4.1-cuda12.4-cudnn9

# ---------------------------------------------------------------- stage 1 ---
FROM pytorch/pytorch:${PYTORCH_TAG}-devel AS builder

# GPU architectures the extension is compiled for (no GPU is present at build time):
#   8.0 = A100 / A30      8.6 = RTX 3090 / A5000 / A6000 / A40
#   8.9 = RTX 4090 / L4 / L40 / L40S      9.0 = H100 / H200   (+PTX = JIT fallback)
# RTX 5090 / B200 (12.0) need a CUDA 12.8+ base image: change PYTORCH_TAG and add "12.0".
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# The NVIDIA apt repos baked into the CUDA images break apt-get update from time to time
# (stale keys / missing Release file). We do not need them: remove them before installing.
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list \
 && apt-get update -o Acquire::Retries=3 \
 && apt-get install -y --no-install-recommends git ninja-build \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 https://github.com/XiShuFan/CrossTooth_CVPR2025.git \
 && git clone --depth 1 https://github.com/POSTECH-CVLab/point-transformer.git

# pointops: <THC/THC.h> no longer exists in modern PyTorch -> drop the include (same fix as the notebook),
# then build the extension as a wheel so the runtime stage does not need nvcc.
WORKDIR /build/point-transformer/lib/pointops
RUN sed -i 's|#include <THC/THC.h>||g' src/*/*.cpp \
 && pip install ninja wheel setuptools \
 && pip wheel --no-build-isolation --no-deps -w /wheels . \
 && ls -l /wheels

# Assemble the repo tree exactly like the notebook did
# (CrossTooth imports models.PointTransformer.libs.pointops -- "libs" with an s)
WORKDIR /build
COPY patch_pointops.py /build/patch_pointops.py
RUN mkdir -p CrossTooth_CVPR2025/models/PointTransformer/libs \
 && cp -r point-transformer/lib/pointops CrossTooth_CVPR2025/models/PointTransformer/libs/pointops \
 && rm -rf CrossTooth_CVPR2025/models/PointTransformer/libs/pointops/build \
           CrossTooth_CVPR2025/models/PointTransformer/libs/pointops/*.egg-info \
 && touch CrossTooth_CVPR2025/models/PointTransformer/__init__.py \
          CrossTooth_CVPR2025/models/PointTransformer/libs/__init__.py \
 && python patch_pointops.py CrossTooth_CVPR2025/models/PointTransformer/libs/pointops/functions/pointops.py \
 && rm -rf CrossTooth_CVPR2025/.git CrossTooth_CVPR2025/compete CrossTooth_CVPR2025/YBSESUN6_upper.obj

# ---------------------------------------------------------------- stage 2 ---
FROM pytorch/pytorch:${PYTORCH_TAG}-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CROSSTOOTH_DIR=/app/CrossTooth_CVPR2025 \
    WARMUP=1

# shared libraries VTK (vedo) may dlopen even when running headless
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list \
 && apt-get update -o Acquire::Retries=3 \
 && apt-get install -y --no-install-recommends \
      libgl1 libxrender1 libxext1 libsm6 libice6 libx11-6 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# compiled pointops_cuda extension (same python + torch as the builder)
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels \
 && python -c "import torch, pointops_cuda; print('pointops_cuda OK')"

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
 && python -c "import torch, vedo, sklearn, runpod; print('torch', torch.__version__, 'vedo', vedo.__version__)"

COPY --from=builder /build/CrossTooth_CVPR2025 /app/CrossTooth_CVPR2025
COPY crosstooth_pipeline.py handler.py test_input.json /app/

# import-time smoke test (no GPU here: only checks that the modules resolve)
RUN cd /app/CrossTooth_CVPR2025 && python -c "from models.PTv1.point_transformer_seg import PointTransformerSeg38; print('CrossTooth imports OK')"

CMD ["python", "-u", "/app/handler.py"]