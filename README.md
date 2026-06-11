# AoE-Repro — Mac-runnable reproduction of the AoE cloud labeling pipeline

A faithful, **Apple-Silicon-runnable** re-implementation of the six-stage cloud
labeling pipeline from *AoE: Always-on Egocentric Human Video Collection for
Embodied AI* ([arXiv 2602.23893](https://arxiv.org/abs/2602.23893)). It turns a
raw egocentric clip into training-ready labels — **3D hand joints (world coords),
6-DoF camera poses, monocular depth, and atomic-action segments** — with each
heavy model swapped for a lightweight open-source substitute that runs on a Mac
(MPS/CPU, no CUDA).

📖 中文说明见 [README.zh-CN.md](README.zh-CN.md)

> **Scope.** The paper is a *data-collection + curation system* (neck-mount
> hardware + mobile app + cloud pipeline + a robot-policy proof) and ships **no
> code, dataset, or models**. This repo reproduces the **cloud labeling pipeline**
> only. The hardware holder, the iOS/Android app, and the Unitree-G1 downstream
> need physical equipment and are out of scope. See [Fidelity](#fidelity-caveats).

## Pipeline: paper model → Mac substitute

| # | Stage | Paper (heavy) | This repo (Mac-runnable) |
|---|-------|---------------|--------------------------|
| 1 | Calibration | Camera2 intrinsics | metadata sidecar → OpenCV checkerboard → FOV pinhole |
| 2 | Action segmentation | Qwen3-VL-235B-A22B | adaptive keyframe-count selection + local/hosted VLM backend; optical-flow heuristic fallback |
| 3 | Camera trajectory + depth | MegaSAM + Lingbot-Depth | ORB monocular VO + **Depth-Anything-V2-Small** (MPS) |
| 4 | Hand reconstruction | HaWoR + MANO | **MediaPipe HandLandmarker** → depth back-projection → world-lift → smoothing |
| 5 | Augmentation (optional) | Masquerade GAN + video diffusion | Selfie-Segmentation bg-swap + `cv2.inpaint` (off by default) |
| 6 | Quality control | 3σ velocity + 5px reproj | NumPy velocity/reprojection filters + 5% manual-inspect sample |
| + | Eval | — | MPJPE / PA-MPJPE / AUC(PCK); ATE / ATE-S / RPE via `evo` |

Execution DAG: `ingest → segment → trajectory → hands → qc` (augment optional),
matching the paper cloud labeling order.

## Setup

Requires Python 3.11/3.12 (MediaPipe/Depth-Anything wheels lag newer Python) and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .          # core pipeline
# optional extras:
uv pip install -e '.[vlm]'       # hosted-VLM segmentation backend
uv pip install -e '.[augment]'   # heavy diffusion augmentation path
uv pip install -e '.[download]'  # dataset downloaders
```

First run downloads two small models automatically: Depth-Anything-V2-Small
(HF cache) and `hand_landmarker.task` (`~/.cache/aoe_pipeline/`).

## Quickstart

```bash
# 1. self-contained synthetic clip (no hand): exercises depth/VO/segment/QC
python scripts/make_sample_clip.py                       # -> data/sample_clip.mp4
aoe-pipeline run --video data/sample_clip.mp4 --output-dir output --verbose

# 2. real-hand demo: pan/zoom a hand photo into an egocentric-style clip
curl -sL -o datasets/hand.jpg \
  https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg
python scripts/make_hand_clip.py --image datasets/hand.jpg --out datasets/hand_clip.mp4
aoe-pipeline run --video datasets/hand_clip.mp4 --output-dir output --verbose

aoe-pipeline list-stages
```

Each run writes `output/<clip_id>/`:

```
manifest.json          intrinsics, fps, per-stage status
frames/frame_%06d.png  decoded RGB
depth/depth_%06d.npy   monocular depth (float32, pseudo-metric)
trajectory.tum         camera poses (TUM; feed to evo)
hands/joints_world.npy (T,2,21,3) 3D joints, world coords (slot 0=Left,1=Right)
hands/joints_2d.npy    (T,2,21,2) image-pixel joints
hands/joints_cam.npy   (T,2,21,3) camera-frame joints
segments.json          atomic-action segments
qc_report.json         flagged frames + pass/fail + manual-inspect sample
viz/                   hand overlays + trajectory plot
```

## Configuration

`configs/default.yaml` mirrors the built-in defaults. Run a custom config with
`--config configs/mine.yaml`, or a subset of stages with `--only ingest,hands`.
Key knobs: `segment.backend` (`heuristic`|`vlm`), `segment.vlm.provider`
(`local_openai`|`ollama`|`anthropic`), `segment.vlm.keyframe_strategy`
(`adaptive`|`fixed`), `hands.smooth_window`, `hands.depth_anchor`
(`wrist`|`per_joint`), `trajectory.depth_model`, `qc.{velocity_sigma,reproj_px}`.
The stage registry (`stages/registry.py`) lets a GPU box swap a substitute for
the real model by adding a new stage class — the orchestrator is untouched.

For the paper-faithful local VLM path, run a vision model behind an
OpenAI-compatible endpoint and set:

```yaml
stages:
  segment:
    params:
      backend: vlm
      vlm:
        provider: local_openai
        model: Qwen2.5-VL-7B-Instruct
        base_url: http://localhost:8000/v1
        keyframe_strategy: adaptive
        min_keyframes: 8
        max_keyframes: 32
        seconds_per_keyframe: 2.0
```

`provider: ollama` is also supported for local Ollama vision models. VLM runs
write `segments.json`, paper-style `atomic_actions.json`, and
`cloud_selection.json` with the selected frame count and indices.

### Paper-Faithful External Backends

`configs/paper_original.yaml` is a wiring template for the original paper
methods:

- **Depth + trajectory:** LingBot-Depth + MegaSAM.
- **Hands:** HaWoR + MANO.
- **Augmentation:** Masquerade / Phantom-style robotization.

These projects require their own CUDA/conda environments, model weights, and in
the case of MANO, separately licensed model files. This repo therefore treats
them as external operators: configure a command, write outputs under
`{external_dir}`, and the AoE pipeline imports the artifacts into its standard
layout. For example, `trajectory.backend: original` expects:

```text
{external_dir}/depth/depth_000000.npy ...
{external_dir}/trajectory.tum
```

`hands.backend: original` expects HaWoR/MANO arrays:

```text
{external_dir}/joints_world.npy  # (T,2,21,3)
{external_dir}/joints_2d.npy     # (T,2,21,2)
```

`augment.backend: original` imports either augmented frames or a Masquerade
overlay video. See `configs/paper_original.yaml` for every placeholder and file
contract.

## Evaluation

```bash
# camera trajectory vs GT (Sim3 / SE3 / RPE)
aoe-pipeline eval-traj --est output/<clip>/trajectory.tum --gt <gt>.tum

# hand pose vs GT (MPJPE / PA-MPJPE / AUC), with optional joint remap
aoe-pipeline eval-hands --pred output/<clip>/hands/joints_world.npy --gt <gt>.npy --to-mano
```

For ground truth, **EgoDex** (Apple) is the lowest-friction benchmark — it ships
3D hand-joint GT. `python scripts/download_egodex.py` prints access steps;
`scripts/download_ego4d.py` covers the license-gated Ego4D flow.

## Visualization

```bash
# segment timeline / contact sheet / annotated video + cut interaction clips
python scripts/visualize_segments.py --clip-dir output/<clip> --source <video>.mp4

# HaWoR-style Front|Top|Side orthographic animation of the 3D hands
python scripts/render_hand_views.py --clip-dir output/<clip> --frame both

# HaWoR-style perspective "world scene": ground plane + ghost-hand trail +
# camera path + frustum, two viewpoints (emulated — hands are hull-rendered
# from 21 joints, not a MANO mesh; world layout is up-to-scale)
python scripts/render_hand_views.py --clip-dir output/<clip> --frame scene
```

`render_hand_views.py` produces `viz/hand_views_camera*.mp4` (hand pose from new
angles; `--camera-depth per_joint` shows finger thickness, `saved` is the planar
wrist-anchored reconstruction), `viz/hand_views_world.mp4` (Front/Top/Side
orthographic, hands + camera trail), and `viz/hand_views_scene.mp4` (the
perspective HaWoR-style world scene).

## Tests

```bash
pytest -q                 # fast unit tests (schema, QC math, metrics, ingest)
AOE_RUN_E2E=1 pytest -q    # + full real-model pipeline on a synthetic clip
```

## Fidelity caveats

This is a working prototype, **not** a paper-accuracy reproduction:

- **Trajectory is scale-ambiguous.** Monocular ORB VO recovers pose up to scale;
  translation is scaled by a fraction of median depth. Evaluate with Sim(3)
  (7-DoF) and scale-free SE(3) alignment — both provided by `eval-traj`.
- **Depth is relative**, not metric (Depth-Anything-V2 outputs affine-invariant
  inverse depth; we map it to a pseudo-metric range).
- **Hand reconstruction** substitutes a 21-point detector + monocular depth for
  HaWoR's MANO mesh fit. A consequence: the back-projection is reprojection-exact
  *before* smoothing, so the QC **reprojection** term mainly flags frames where a
  jittery detection disagrees with the smoothed estimate (it does not measure
  mesh-fit quality as in the paper). The **velocity** filter is the substantive
  temporal check. Expect reprojection well above the paper's 5px on
  low-resolution / fast-motion clips.
- **Augmentation** is a lightweight segmentation+inpaint stand-in for the paper's
  GAN/diffusion pipeline.

Every stage exposes the same interface as its faithful counterpart, so swapping
in the real models (MegaSAM, HaWoR, a 235B VLM) on a GPU box is a config/registry
change, not a rewrite.

## What is **not** reproduced

The neck-mount hardware, the on-device mobile app + selective recording, and the
downstream GR00T-N1.5 / Unitree-G1 robot policy — all require physical equipment.

## Disclaimer & data

- This is an **independent reproduction** for research/educational purposes, **not
  affiliated with or endorsed by** the authors of the AoE paper. It re-implements the
  pipeline's architecture with lightweight substitute models; it is not a
  paper-accuracy or official release.
- Third-party models (MediaPipe, Depth-Anything-V2, etc.) are downloaded at runtime
  under their own licenses and are **not** redistributed here.
- **No personal or third-party media is committed.** Any real recorded video and all
  generated outputs (`output/`, `datasets/`) are git-ignored. Only the **synthetic**
  `data/sample_clip.mp4` is bundled so the quickstart runs out of the box; build a
  hand demo clip yourself via `scripts/make_hand_clip.py`.
- The code is released under the [MIT License](LICENSE).
