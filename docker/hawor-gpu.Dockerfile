FROM hub.byted.org/base/cuda-12.9-ubuntu22.04-py3.11:latest

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://bytedpypi.byted.org/simple/ \
    UV_INDEX_URL=https://bytedpypi.byted.org/simple/ \
    UV_LINK_MODE=copy \
    TORCH_CUDA_ARCH_LIST=9.0 \
    MAX_JOBS=8 \
    AOE_ROOT=/opt/aoe-repo \
    EXTERNAL_ROOT=/opt/aoe-repo/external \
    HAWOR_ROOT=/opt/aoe-repo/external/HaWoR \
    PYTORCH3D_ROOT=/opt/aoe-repo/external/pytorch3d \
    AOE_VENV=/opt/aoe-repo/.venv \
    PATH=/root/.local/bin:/opt/aoe-repo/.venv/bin:${PATH}

SHELL ["/bin/bash", "-lc"]

ARG PYPI_INDEX=https://bytedpypi.byted.org/simple/
ARG TORCH_VERSION=2.4.0
ARG TORCHVISION_VERSION=0.19.0
ARG HAWOR_ZIP=https://github.com/ThunderVVV/HaWoR/archive/refs/heads/main.zip
ARG PYTORCH3D_ZIP=https://github.com/facebookresearch/pytorch3d/archive/refs/heads/main.zip

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libglvnd0 \
      libsm6 \
      libx11-6 \
      libxext6 \
      libxrender1 \
      ninja-build \
      pkg-config \
      python3-dev \
      python3-pip \
      unzip \
      wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --user -i "${PYPI_INDEX}" uv

WORKDIR /opt

COPY external/ /tmp/external-zips/
COPY docker/source-cache/ /tmp/source-cache/

RUN set -eux; \
    download_zip() { \
      local url="$1"; \
      local dst="$2"; \
      rm -f "${dst}"; \
      curl --fail --location --http1.1 \
        --retry 10 --retry-delay 5 --retry-all-errors \
        --connect-timeout 30 --speed-time 60 --speed-limit 1024 \
        "${url}" -o "${dst}"; \
    }; \
    source_tree_or_zip() { \
      local local_dir="$1"; \
      local local_zip="$2"; \
      local url="$3"; \
      local target_dir="$4"; \
      local tmp_zip="$5"; \
      local top_prefix="$6"; \
      if [[ -f "${local_dir}/requirements.txt" || -f "${local_dir}/setup.py" ]]; then \
        shopt -s dotglob; \
        cp -a "${local_dir}/"* "${target_dir}/"; \
      else \
        if [[ -s "${local_zip}" ]]; then \
          cp "${local_zip}" "${tmp_zip}"; \
        else \
          download_zip "${url}" "${tmp_zip}"; \
        fi; \
        unzip -q "${tmp_zip}" -d /tmp; \
        shopt -s dotglob; \
        mv /tmp/"${top_prefix}"-*/* "${target_dir}/"; \
        rm -rf "${tmp_zip}" /tmp/"${top_prefix}"-*; \
      fi; \
    }; \
    mkdir -p "${HAWOR_ROOT}" "${PYTORCH3D_ROOT}"; \
    source_tree_or_zip /tmp/source-cache/HaWoR /tmp/external-zips/hawor.zip \
      "${HAWOR_ZIP}" "${HAWOR_ROOT}" /tmp/hawor.zip HaWoR; \
    source_tree_or_zip /tmp/source-cache/pytorch3d /tmp/external-zips/pytorch3d.zip \
      "${PYTORCH3D_ZIP}" "${PYTORCH3D_ROOT}" /tmp/pytorch3d.zip pytorch3d

COPY . ${AOE_ROOT}

RUN rm -rf "${AOE_ROOT}/docker/source-cache" "${AOE_ROOT}/external/"*.zip

RUN uv venv --python 3.11 "${AOE_VENV}" \
    && uv pip install --python "${AOE_VENV}/bin/python" \
      --index-url "${PYPI_INDEX}" \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
    && uv pip install --python "${AOE_VENV}/bin/python" \
      --index-url "${PYPI_INDEX}" \
      pip \
      setuptools \
      wheel \
      ninja \
      packaging \
    && uv pip install --python "${AOE_VENV}/bin/python" \
      --index-url "${PYPI_INDEX}" \
      -e "${AOE_ROOT}[dev,render]"

WORKDIR ${HAWOR_ROOT}

RUN grep -v 'pytorch3d' requirements.txt \
      | grep -v 'chumpy@git' \
      | grep -v '^chumpy' \
      | grep -v 'torch-scatter' \
      > /tmp/hawor_requirements_no_git.txt \
    && uv pip install --python "${AOE_VENV}/bin/python" \
      --index-url "${PYPI_INDEX}" \
      numpy==1.26.4 \
      -r /tmp/hawor_requirements_no_git.txt \
      pytorch-lightning==2.2.4 \
      lightning-utilities \
      torchmetrics==1.4.0 \
    && uv pip install --python "${AOE_VENV}/bin/python" \
      --index-url "${PYPI_INDEX}" \
      --no-build-isolation \
      chumpy==0.70 \
    && uv pip install --python "${AOE_VENV}/bin/python" \
      --index-url "${PYPI_INDEX}" \
      --no-build-isolation \
      torch-scatter==2.1.2

RUN if ! command -v nvcc >/dev/null 2>&1; then \
      echo "ERROR: nvcc is required to build PyTorch3D and DROID-SLAM CUDA extensions."; \
      echo "Use a CUDA devel base image or install CUDA toolkit/nvcc in the base image."; \
      exit 2; \
    fi

RUN uv pip install --python "${AOE_VENV}/bin/python" \
      --index-url "${PYPI_INDEX}" \
      --no-build-isolation \
      "${PYTORCH3D_ROOT}"

WORKDIR ${HAWOR_ROOT}/thirdparty/DROID-SLAM

RUN "${AOE_VENV}/bin/python" setup.py install

RUN mkdir -p \
      "${HAWOR_ROOT}/_DATA/data/mano" \
      "${HAWOR_ROOT}/_DATA/data_left/mano_left" \
      "${HAWOR_ROOT}/weights/external" \
      "${HAWOR_ROOT}/weights/hawor/checkpoints" \
      "${HAWOR_ROOT}/thirdparty/Metric3D/weights" \
      /workspace/output

RUN "${AOE_VENV}/bin/python" - <<'PY'
import importlib.util
import torch

mods = [
    "torch",
    "torchvision",
    "cv2",
    "pyrender",
    "smplx",
    "mmcv",
    "torch_scatter",
    "pytorch3d",
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("HaWoR torch:", torch.__version__, "cuda:", torch.version.cuda)
if missing:
    raise SystemExit(f"missing HaWoR imports: {missing}")
PY

RUN "${AOE_ROOT}/.venv/bin/python" - <<'PY'
import aoe_pipeline
print("AoE import ok:", aoe_pipeline.__file__)
PY

WORKDIR ${AOE_ROOT}

CMD ["/bin/bash"]
