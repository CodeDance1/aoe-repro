# HaWoR GPU Docker Image

Build from the ByteDance CUDA base image:

```bash
docker build -f docker/hawor-gpu.Dockerfile -t aoe-hawor-gpu:cu129 .
```

The Dockerfile copies source archives from the repo build context first:

```text
external/hawor.zip
external/pytorch3d.zip
```

These zip files are kept in git so the build does not need GitHub access for
HaWoR or PyTorch3D. During `docker build`, the image unzips them into:

```text
/opt/aoe-repo/external/HaWoR
/opt/aoe-repo/external/pytorch3d
```

The Dockerfile can also use local source directories in:

```text
docker/source-cache/HaWoR/
docker/source-cache/pytorch3d/
```

If a source directory exists, it takes priority over the repo zip for that
project. If both the source directory and repo zip are missing, the build
downloads the archives with HTTP/1.1 and retry settings. The expected zip
contents are the standard GitHub source archives with top-level directories
like `HaWoR-main/` and `pytorch3d-main/`.

If the repo zips need to be refreshed:

```bash
mkdir -p external
curl -L --http1.1 https://github.com/ThunderVVV/HaWoR/archive/refs/heads/main.zip \
  -o external/hawor.zip
curl -L --http1.1 https://github.com/facebookresearch/pytorch3d/archive/refs/heads/main.zip \
  -o external/pytorch3d.zip
docker build -f docker/hawor-gpu.Dockerfile -t aoe-hawor-gpu:cu129 .
```

If you already have unpacked source trees:

```bash
mkdir -p docker/source-cache
cp -a /path/to/HaWoR docker/source-cache/HaWoR
cp -a /path/to/pytorch3d docker/source-cache/pytorch3d
docker build -f docker/hawor-gpu.Dockerfile -t aoe-hawor-gpu:cu129 .
```

On the current GPU host, the helper below copies the existing external source
trees into `docker/source-cache/` and excludes weights, MANO files, generated
arrays, videos, and `.git` directories:

```bash
./scripts/prepare_hawor_docker_source_cache.sh
docker build -f docker/hawor-gpu.Dockerfile -t aoe-hawor-gpu:cu129 .
```

The image installs:

- AoE and HaWoR dependencies into one uv environment:
  `/opt/aoe-repo/.venv` with Python 3.11.
- HaWoR source into `/opt/aoe-repo/external/HaWoR`.
- CUDA/PyTorch dependencies for H100, PyTorch3D, torch-scatter, and DROID-SLAM.
- System libraries needed for OpenCV, pyrender/OpenGL/EGL, ffmpeg, and CUDA extension builds.

The image intentionally does **not** include MANO files or large model weights.
Place or mount them at:

```text
/opt/aoe-repo/external/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
/opt/aoe-repo/external/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
/opt/aoe-repo/external/HaWoR/weights/external/detector.pt
/opt/aoe-repo/external/HaWoR/weights/external/droid.pth
/opt/aoe-repo/external/HaWoR/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
/opt/aoe-repo/external/HaWoR/weights/hawor/checkpoints/hawor.ckpt
/opt/aoe-repo/external/HaWoR/weights/hawor/checkpoints/infiller.pt
/opt/aoe-repo/external/HaWoR/weights/hawor/model_config.yaml
```

Run with GPU access and mount data/output:

```bash
docker run --gpus all --rm -it \
  -v /path/to/hawor_weights:/opt/aoe-repo/external/HaWoR/weights \
  -v /path/to/mano_right:/opt/aoe-repo/external/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl:ro \
  -v /path/to/mano_left:/opt/aoe-repo/external/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl:ro \
  -v "$PWD/data":/workspace/data \
  -v "$PWD/output":/workspace/output \
  aoe-hawor-gpu:cu129
```

Inside the container:

```bash
python /opt/aoe-repo/scripts/check_hawor_runtime_gpu.py

aoe-pipeline run \
  --video /workspace/data/sample_clip.mp4 \
  --output-dir /workspace/output \
  --config /opt/aoe-repo/configs/hawor_hybrid_docker.yaml \
  --verbose

aoe-pipeline render-hawor-demo \
  --clip-dir /workspace/output/sample_clip \
  --out /workspace/output/sample_clip/viz/hawor_demo.mp4 \
  --fps 30 \
  --size 720 \
  --prefer-mesh true

aoe-pipeline check-hawor-demo \
  --clip-dir /workspace/output/sample_clip \
  --demo-mp4 /workspace/output/sample_clip/viz/hawor_demo.mp4 \
  --require-mesh true \
  --expected-size 720
```

If the build fails at the `nvcc` check, the base image is a runtime image rather
than a devel image. Use a CUDA devel base image, or extend the base image with a
CUDA toolkit package that provides `nvcc`.
