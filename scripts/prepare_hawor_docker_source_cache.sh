#!/usr/bin/env bash
set -euo pipefail

# Prepare Docker build-context source cache for HaWoR/PyTorch3D.
#
# This script does not install dependencies, download packages, or build images.
# It only copies source trees into docker/source-cache/ so docker build can avoid
# fetching source archives from GitHub.

WORK_ROOT="${WORK_ROOT:-/mnt/bn/tiktok-mm-5/mlx/users/yanlin.chn}"
AOE_ROOT="${AOE_ROOT:-$WORK_ROOT/repo/aoe-repro}"
HAWOR_SRC="${HAWOR_SRC:-$WORK_ROOT/external/HaWoR}"
PYTORCH3D_SRC="${PYTORCH3D_SRC:-$WORK_ROOT/external/pytorch3d}"
SOURCE_CACHE="${SOURCE_CACHE:-$AOE_ROOT/docker/source-cache}"

copy_source_tree() {
  local src="$1"
  local dst="$2"
  local name="$3"

  if [[ ! -d "$src" ]]; then
    echo "MISSING: $name source directory: $src"
    return 1
  fi

  rm -rf "$dst"
  mkdir -p "$dst"

  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '.git/' \
      --exclude '__pycache__/' \
      --exclude '*.pyc' \
      --exclude '*.mp4' \
      --exclude '*.mov' \
      --exclude '*.avi' \
      --exclude '*.mkv' \
      --exclude '*.npy' \
      --exclude '*.npz' \
      --exclude '*.pth' \
      --exclude '*.pt' \
      --exclude '*.ckpt' \
      --exclude '*.pkl' \
      --exclude 'weights/' \
      --exclude '_DATA/' \
      "$src"/ "$dst"/
  else
    tar \
      --exclude='.git' \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='*.mp4' \
      --exclude='*.mov' \
      --exclude='*.avi' \
      --exclude='*.mkv' \
      --exclude='*.npy' \
      --exclude='*.npz' \
      --exclude='*.pth' \
      --exclude='*.pt' \
      --exclude='*.ckpt' \
      --exclude='*.pkl' \
      --exclude='weights' \
      --exclude='_DATA' \
      -C "$src" -cf - . | tar -C "$dst" -xf -
  fi
}

echo "==> AoE root: $AOE_ROOT"
echo "==> Source cache: $SOURCE_CACHE"
mkdir -p "$SOURCE_CACHE"

copy_source_tree "$HAWOR_SRC" "$SOURCE_CACHE/HaWoR" "HaWoR"
copy_source_tree "$PYTORCH3D_SRC" "$SOURCE_CACHE/pytorch3d" "PyTorch3D"

echo "==> Docker source cache prepared:"
du -sh "$SOURCE_CACHE/HaWoR" "$SOURCE_CACHE/pytorch3d"
