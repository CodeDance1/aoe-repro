#!/usr/bin/env bash
set -euo pipefail

# Install a uv-managed HaWoR runtime environment for the AoE hybrid pipeline.
#
# This script installs Python/CUDA dependencies only. MANO files are
# license-gated and must be placed manually. Large model weights are checked at
# the end but are not downloaded here.

PYPI_INDEX="${PYPI_INDEX:-https://bytedpypi.byted.org/simple/}"
WORK_ROOT="${WORK_ROOT:-/mnt/bn/tiktok-mm-5/mlx/users/yanlin.chn}"
AOE_ROOT="${AOE_ROOT:-$WORK_ROOT/repo/aoe-repro}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$WORK_ROOT/external}"
HAWOR_ROOT="${HAWOR_ROOT:-$EXTERNAL_ROOT/HaWoR}"
PYTORCH3D_ROOT="${PYTORCH3D_ROOT:-$EXTERNAL_ROOT/pytorch3d}"
AOE_VENV="${AOE_VENV:-$AOE_ROOT/.venv}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.0}"

export UV_INDEX_URL="$PYPI_INDEX"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

echo "==> Using index: $PYPI_INDEX"
echo "==> AoE root: $AOE_ROOT"
echo "==> HaWoR root: $HAWOR_ROOT"
echo "==> uv environment: $AOE_VENV"

if [[ ! -x "$UV_BIN" ]]; then
  echo "==> Installing uv to $UV_BIN"
  python -m pip install --user -i "$PYPI_INDEX" uv
fi

mkdir -p "$EXTERNAL_ROOT"

if [[ ! -d "$HAWOR_ROOT" ]]; then
  echo "==> HaWoR repo not found. Downloading source archive."
  tmp_zip="$EXTERNAL_ROOT/hawor-main.zip"
  rm -rf "$tmp_zip" "$EXTERNAL_ROOT/HaWoR-main"
  python - <<PY
from urllib.request import urlretrieve
urlretrieve(
    "https://github.com/ThunderVVV/HaWoR/archive/refs/heads/main.zip",
    "$tmp_zip",
)
PY
  python -m zipfile -e "$tmp_zip" "$EXTERNAL_ROOT"
  mv "$EXTERNAL_ROOT/HaWoR-main" "$HAWOR_ROOT"
fi

if [[ ! -x "$AOE_VENV/bin/python" ]]; then
  echo "==> Creating uv venv with Python $PYTHON_VERSION"
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$AOE_VENV"
fi

PY="$AOE_VENV/bin/python"
ACTUAL_PYTHON_VERSION="$("$PY" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "$ACTUAL_PYTHON_VERSION" != "$PYTHON_VERSION" ]]; then
  echo "==> Existing uv environment uses Python $ACTUAL_PYTHON_VERSION, expected $PYTHON_VERSION."
  echo "    Remove or recreate $AOE_VENV with Python $PYTHON_VERSION, then rerun this script."
  exit 2
fi

echo "==> Installing PyTorch $TORCH_VERSION / torchvision $TORCHVISION_VERSION"
"$UV_BIN" pip install --python "$PY" --index-url "$PYPI_INDEX" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION"

echo "==> Installing build helpers"
"$UV_BIN" pip install --python "$PY" --index-url "$PYPI_INDEX" \
  pip \
  setuptools \
  wheel \
  ninja \
  packaging

echo "==> Installing AoE package into the shared uv environment"
"$UV_BIN" pip install --python "$PY" --index-url "$PYPI_INDEX" \
  -e "$AOE_ROOT[dev,render]"

cd "$HAWOR_ROOT"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "==> nvcc is not on PATH."
  echo "    torch-scatter, PyTorch3D, and DROID-SLAM build CUDA extensions."
  echo "    Load/install a CUDA toolkit first, then rerun this script."
  exit 2
fi

echo "==> Installing HaWoR Python requirements from internal mirror"
grep -v 'pytorch3d' requirements.txt \
  | grep -v 'chumpy@git' \
  | grep -v '^chumpy' \
  | grep -v 'torch-scatter' \
  > /tmp/hawor_requirements_no_git.txt
"$UV_BIN" pip install --python "$PY" --index-url "$PYPI_INDEX" \
  numpy==1.26.4 \
  -r /tmp/hawor_requirements_no_git.txt \
  pytorch-lightning==2.2.4 \
  lightning-utilities \
  torchmetrics==1.4.0

echo "==> Installing chumpy with no build isolation"
"$UV_BIN" pip install --python "$PY" --index-url "$PYPI_INDEX" \
  --no-build-isolation \
  chumpy==0.70

echo "==> Installing torch-scatter with no build isolation"
"$UV_BIN" pip install --python "$PY" --index-url "$PYPI_INDEX" \
  --no-build-isolation \
  torch-scatter==2.1.2

if [[ ! -d "$PYTORCH3D_ROOT" ]]; then
  echo "==> PyTorch3D source not found at $PYTORCH3D_ROOT"
  echo "    Clone or unpack PyTorch3D there, then rerun this script:"
  echo "    git clone https://github.com/facebookresearch/pytorch3d.git $PYTORCH3D_ROOT"
  exit 2
fi

echo "==> Installing PyTorch3D from local source: $PYTORCH3D_ROOT"
"$UV_BIN" pip install --python "$PY" --index-url "$PYPI_INDEX" \
  --no-build-isolation \
  "$PYTORCH3D_ROOT"

echo "==> Building/installing DROID-SLAM CUDA extension"
(
  cd "$HAWOR_ROOT/thirdparty/DROID-SLAM"
  "$PY" setup.py install
)

echo "==> Verifying key Python imports and H100 CUDA availability"
"$PY" - <<'PY'
import importlib
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
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
if missing:
    raise SystemExit(f"missing imports after install: {missing}")
PY

echo "==> Checking MANO and model weight files"
missing=0
for path in \
  "$HAWOR_ROOT/_DATA/data/mano/MANO_RIGHT.pkl" \
  "$HAWOR_ROOT/_DATA/data_left/mano_left/MANO_LEFT.pkl" \
  "$HAWOR_ROOT/weights/external/detector.pt" \
  "$HAWOR_ROOT/weights/external/droid.pth" \
  "$HAWOR_ROOT/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth" \
  "$HAWOR_ROOT/weights/hawor/checkpoints/hawor.ckpt" \
  "$HAWOR_ROOT/weights/hawor/checkpoints/infiller.pt" \
  "$HAWOR_ROOT/weights/hawor/model_config.yaml"; do
  if [[ -f "$path" ]]; then
    ls -lh "$path"
  else
    echo "MISSING: $path"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Python dependencies are installed, but MANO/model files are incomplete."
  echo "Place the missing files in the paths above, then tell Codex to continue."
  exit 3
fi

echo "==> HaWoR uv environment is ready."
echo "    Python: $PY"
echo "    HAWOR_ROOT=$HAWOR_ROOT"
