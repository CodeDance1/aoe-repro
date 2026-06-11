#!/usr/bin/env bash
# Set up HaWoR (faithful Stage 4) in its OWN conda env on a CUDA box.
# HaWoR needs py3.10 / torch1.13 / cu117 — incompatible with the base env, hence
# the subprocess-adapter design (the `hands_hawor` stage shells into this env).
#
# Usage:  bash envs/hawor.sh [TARGET_DIR]    (default: ./third_party/HaWoR)
set -euo pipefail

TARGET="${1:-third_party/HaWoR}"
ENV_NAME="hawor"

echo "== 1/4 clone HaWoR =="
if [ ! -d "$TARGET" ]; then
  git clone --recursive https://github.com/ThunderVVV/HaWoR "$TARGET"
fi

echo "== 2/4 conda env ($ENV_NAME: py3.10 torch1.13 cu117) =="
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.10
  conda run -n "$ENV_NAME" pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
  conda run -n "$ENV_NAME" pip install -r "$TARGET/requirements.txt"
  conda run -n "$ENV_NAME" pip install pytorch-lightning==2.2.4 torchmetrics==1.4.0 scipy
fi

echo "== 3/4 checkpoints =="
mkdir -p "$TARGET/weights/external" "$TARGET/weights/hawor/checkpoints" \
         "$TARGET/thirdparty/Metric3D/weights"
cat <<'EOF'
Download per the HaWoR README (https://github.com/ThunderVVV/HaWoR#installation):
  droid.pth                        -> weights/external/
  detector.pt                      -> weights/external/
  hawor.ckpt, infiller.pt          -> weights/hawor/checkpoints/
  model_config.yaml                -> weights/hawor/
  metric_depth_vit_large_800k.pth  -> thirdparty/Metric3D/weights/
EOF

echo "== 4/4 MANO models (license-gated) =="
cat <<'EOF'
Register at https://mano.is.tue.mpg.de/ , download MANO_LEFT.pkl + MANO_RIGHT.pkl,
place under: <HaWoR>/_DATA/data/mano/   (see HaWoR README for the exact dir).

Then point configs/faithful.yaml stages.hands_hawor.params at:
  hawor_dir: <absolute path to the HaWoR checkout>
  conda_env: hawor
  mano_dir:  <HaWoR>/_DATA/data/mano
Smoke test:
  conda run -n hawor python demo.py --video_path ./example/video_0.mp4 --vis_mode world
Then verify everything from the repo root:
  python scripts/check_hawor_env.py --hawor-dir TARGET --check-imports
EOF
echo "done."
