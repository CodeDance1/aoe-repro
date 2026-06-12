# HaWoR + MediaPipe Hybrid GPU Migration Plan

This plan is for running the AoE reproduction pipeline on a GPU machine and
upgrading the hand reconstruction stage from the current lightweight MediaPipe
fallback to a hybrid pipeline:

1. MediaPipe provides fast 2D hand detection, hand boxes, handedness, and a
   robust fallback.
2. HaWoR + MANO provides higher-quality camera/world-space 3D hand
   reconstruction.
3. The AoE pipeline imports HaWoR outputs into the existing standard files:
   `hands/joints_world.npy`, `hands/joints_cam.npy`, and `hands/joints_2d.npy`.

## Target Outcome

After this migration, a GPU run should support:

```yaml
hands:
  params:
    backend: hybrid
    hybrid:
      detector: mediapipe
      reconstructor: hawor
      use_mediapipe_bbox: true
      use_mediapipe_landmarks: true
      fallback_to_mediapipe: true
```

Expected behavior:

- Run MediaPipe first for all frames.
- Export MediaPipe hand hints for HaWoR:
  - hand boxes
  - 2D landmarks
  - handedness / slot assignment
  - frame presence mask
- Run HaWoR in the shared uv-managed GPU environment.
- Import HaWoR/MANO results when available.
- Fill failed HaWoR frames with MediaPipe fallback results when configured.
- Record fallback counts and backend metadata in `manifest.json` and
  `hands/meta.json`.

## Why Hybrid Instead of HaWoR Only

HaWoR is the better high-fidelity 3D/world-space reconstructor, but it still
benefits from reliable detection, cropping, handedness, and failure handling.
MediaPipe is fast and robust for those front-end tasks. The hybrid architecture
also makes debugging easier because MediaPipe outputs can be inspected before
HaWoR is invoked.

## Dependencies

### AoE Repo Environment

Use one uv-managed Python 3.11 environment for AoE and HaWoR dependencies:

```bash
cd /path/to/aoe-repro
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev,render]'
```

Run the current tests:

```bash
pytest -q
```

### HaWoR Source And Runtime

Keep HaWoR source outside git-tracked files, or inside the Docker image under
`/opt/aoe-repo/external/HaWoR`. The repository ignores `external/`.

- HaWoR code: https://github.com/ThunderVVV/HaWoR
- HaWoR project page: https://hawor-project.github.io/
- MANO model files: https://mano.is.tue.mpg.de/

Notes:

- Use `scripts/install_hawor_uv_env.sh` on the host, or
  `docker/hawor-gpu.Dockerfile` in a container, to install HaWoR dependencies
  into the same uv environment as AoE.
- The Docker image uses `/opt/aoe-repo/.venv` with Python 3.11 and
  `/opt/aoe-repo/external/HaWoR`.
- MANO files are license-gated. Download them manually and place them where
  HaWoR expects them.
- Do not commit MANO files, model checkpoints, videos, or generated outputs.

Smoke test HaWoR independently before connecting it to AoE:

```bash
cd /path/to/HaWoR
../aoe-repro/.venv/bin/python demo.py --video_path <small_video> --img_focal <fx>
```

Only continue once HaWoR can process a small video or frame sequence by itself.

## Wrapper Contract

Create a wrapper script on the GPU machine, for example:

```text
/path/to/wrappers/run_hawor_with_mediapipe_hints.py
```

The wrapper should accept:

```bash
python run_hawor_with_mediapipe_hints.py \
  --video <video_path> \
  --frames <frames_dir> \
  --trajectory <trajectory_tum> \
  --intrinsics <intrinsics_json> \
  --mediapipe-hints <mediapipe_hints_npz> \
  --out <external_dir>
```

The wrapper must write:

```text
<external_dir>/joints_world.npy  # shape: (T, 2, 21, 3), float32
<external_dir>/joints_2d.npy     # shape: (T, 2, 21, 2), float32
<external_dir>/joints_cam.npy    # optional, shape: (T, 2, 21, 3), float32
<external_dir>/meta.json         # optional, useful for debugging
```

Slot convention:

- slot `0` = left hand
- slot `1` = right hand
- missing values must be `NaN`

Coordinate convention:

- `joints_2d`: pixel coordinates in original frame resolution.
- `joints_cam`: camera-frame 3D joints.
- `joints_world`: world-frame 3D joints aligned with `trajectory.tum`.

## AoE Code Changes To Make On GPU Machine

### 1. Add MediaPipe Hint Export

Refactor the current `HandStage` so the MediaPipe pass can be reused by both
the current substitute backend and the new hybrid backend.

Suggested helper:

```python
def _run_mediapipe_detection(ctx, params) -> dict:
    return {
        "joints_2d": joints_2d,
        "joints_cam": joints_cam,
        "joints_world": joints_world,
        "handedness_per_frame": handed_per_frame,
        "boxes": boxes,
        "presence": presence,
    }
```

Save hints to:

```text
<clip_dir>/hands/mediapipe_hints.npz
```

Recommended arrays:

```text
joints_2d      (T,2,21,2)
joints_cam     (T,2,21,3)
joints_world   (T,2,21,3)
boxes          (T,2,4)      # x1,y1,x2,y2, NaN when missing
presence       (T,2)        # bool
```

### 2. Add `backend: hybrid`

In `src/aoe_pipeline/stages/s4_hands.py`, dispatch:

```python
backend = self.params.get("backend", "substitute")
if backend == "original":
    self._run_original(ctx)
elif backend == "hybrid":
    self._run_hybrid(ctx)
else:
    self._run_substitute(ctx)
```

Suggested `_run_hybrid` flow:

1. Run MediaPipe detection and save `mediapipe_hints.npz`.
2. Run configured HaWoR command via `run_external`.
3. Load HaWoR arrays from `{external_dir}`.
4. Validate shape and frame count.
5. If `fallback_to_mediapipe` is true, fill frames where HaWoR is missing but
   MediaPipe has valid fallback.
6. Save final arrays to `hands/`.
7. Save metadata:
   - `backend: hybrid`
   - `method: MediaPipe+HaWoR+MANO`
   - `hawor_frames`
   - `mediapipe_fallback_frames`
   - `frames_with_hand`

### 3. Add Config

Create a GPU config, for example:

```yaml
pipeline: [ingest, segment, trajectory, hands, qc]

stages:
  ingest:
    enabled: true
    params:
      stride: 1
      max_frames: 0
      hfov_deg: 70.0

  segment:
    enabled: true
    params:
      backend: heuristic

  trajectory:
    enabled: true
    params:
      backend: substitute

  hands:
    enabled: true
    params:
      backend: hybrid
      max_hands: 2
      smooth_window: 5
      smooth_poly: 3
      depth_anchor: wrist
      hybrid:
        use_mediapipe_bbox: true
        use_mediapipe_landmarks: true
        fallback_to_mediapipe: true
        original:
          command:
            - /path/to/aoe-repro/.venv/bin/python
            - /path/to/aoe-repro/scripts/run_hawor_with_mediapipe_hints.py
            - --video
            - "{video_path}"
            - --frames
            - "{frames_dir}"
            - --trajectory
            - "{trajectory_tum}"
            - --intrinsics
            - "{intrinsics_json}"
            - --mediapipe-hints
            - "{mediapipe_hints}"
            - --out
            - "{external_dir}"
          joints_world: joints_world.npy
          joints_2d: joints_2d.npy
          joints_cam: joints_cam.npy

  qc:
    enabled: true
    params:
      velocity_sigma: 3.0
      reproj_px: 5.0
```

### 4. Add Tests

Add tests that do not require a real GPU:

1. Mock the external HaWoR command and write fake arrays.
2. Assert hybrid loads HaWoR outputs.
3. Assert fallback fills missing HaWoR frames from MediaPipe arrays.
4. Assert bad shapes raise `ExternalPipelineError`.

Minimum expected test command:

```bash
pytest -q tests/test_external_adapters.py tests/test_pipeline_smoke.py
```

## Validation Checklist

### Phase 1: Current Pipeline Baseline

```bash
aoe-pipeline run \
  --video data/sample_clip.mp4 \
  --output-dir output_baseline \
  --only ingest,trajectory,hands,qc \
  --verbose
```

Confirm:

```text
output_baseline/<clip>/hands/joints_world.npy
output_baseline/<clip>/hands/joints_2d.npy
output_baseline/<clip>/qc_report.json
```

### Phase 2: HaWoR Wrapper Standalone

Run the wrapper manually outside AoE and inspect outputs:

```bash
/path/to/aoe-repro/.venv/bin/python \
  /path/to/aoe-repro/scripts/run_hawor_with_mediapipe_hints.py \
  --video data/sample_clip.mp4 \
  --frames output_baseline/sample_clip/frames \
  --trajectory output_baseline/sample_clip/trajectory.tum \
  --intrinsics output_baseline/sample_clip/paper_original/hands/intrinsics.json \
  --mediapipe-hints output_baseline/sample_clip/hands/mediapipe_hints.npz \
  --out /tmp/hawor_test
```

Confirm:

```bash
python - <<'PY'
import numpy as np
for name in ["joints_world.npy", "joints_2d.npy"]:
    arr = np.load(f"/tmp/hawor_test/{name}")
    print(name, arr.shape, np.isfinite(arr).mean())
PY
```

### Phase 3: AoE Hybrid Run

```bash
aoe-pipeline run \
  --video data/sample_clip.mp4 \
  --output-dir output_hybrid \
  --config configs/hawor_hybrid_gpu.yaml \
  --verbose
```

Confirm manifest fields:

```bash
cat output_hybrid/sample_clip/manifest.json
cat output_hybrid/sample_clip/hands/meta.json
```

Look for:

```text
backend: hybrid
method: MediaPipe+HaWoR+MANO
hawor_frames
mediapipe_fallback_frames
```

### Phase 4: Visual Check

```bash
aoe-pipeline viz --clip-dir output_hybrid/sample_clip
```

Inspect:

```text
output_hybrid/sample_clip/viz/
```

### Phase 5: QC Comparison

Compare baseline and hybrid:

```bash
cat output_baseline/sample_clip/qc_report.json
cat output_hybrid/sample_clip/qc_report.json
```

Expected improvements:

- fewer high-velocity hand-joint outliers
- more stable hand trajectory in world coordinates
- better temporal continuity through occlusion or motion blur

## HaWoR-Style Demo Video Rendering Plan

The HaWoR project page demo video at
`https://hawor-project.github.io/static/images/page_video.mp4` is a square
720x720, 30 FPS, four-panel visualization. A representative frame shows:

- top-left: raw input video
- top-right: camera-view reconstruction, with colored hand surfaces overlaid on
  the input view
- bottom-left: top view of the reconstructed scene
- bottom-right: side view of the reconstructed scene

The desired AoE output should follow the same layout:

```text
+-------------------------+-------------------------+
| Input video             | Camera view             |
| raw egocentric frame    | hand mesh projected     |
+-------------------------+-------------------------+
| Top view                | Side view               |
| world/camera top-down   | world/camera side view  |
+-------------------------+-------------------------+
```

### Rendering Strategy

Implement this as a new visualization command rather than baking it into the
hand reconstruction stage:

```bash
aoe-pipeline render-hawor-demo \
  --clip-dir output_hybrid/<clip> \
  --out output_hybrid/<clip>/viz/hawor_demo.mp4 \
  --fps 30 \
  --size 720 \
  --prefer-mesh true
```

Why a separate command:

- It can run after either MediaPipe-only or HaWoR-hybrid processing.
- It avoids rerunning heavy HaWoR inference just to tweak visualization style.
- It can support two quality tiers:
  - joint-hull emulation from current AoE outputs
  - real MANO mesh rendering from HaWoR outputs

### Required Inputs

Minimum inputs already available in AoE:

```text
frames/frame_%06d.png
hands/joints_2d.npy
hands/joints_cam.npy
hands/joints_world.npy
trajectory.tum
manifest.json
```

Additional HaWoR/MANO inputs for high-fidelity rendering:

```text
hands/mano_pose.npy       # optional, shape depends on HaWoR wrapper
hands/mano_shape.npy      # optional
hands/mano_trans.npy      # optional
hands/verts_cam.npy       # preferred, (T,2,V,3)
hands/verts_world.npy     # preferred, (T,2,V,3)
hands/faces.npy           # preferred, MANO triangle indices
```

Recommendation: make the HaWoR wrapper export `verts_cam.npy`,
`verts_world.npy`, and `faces.npy` directly. Rendering from vertices is simpler
and less brittle than reconstructing MANO again inside AoE.

### Two-Tier Rendering Plan

#### Tier 1: Current-Capability Joint-Hull Renderer

This can be implemented immediately with current outputs.

Use:

- `joints_2d.npy` for 2D overlay
- `joints_cam.npy` for camera-view 3D hand surface approximation
- `joints_world.npy` + `trajectory.tum` for top/side views

Hand surface approximation:

- For each hand, take the 21 joints.
- Build a convex hull or palm/finger tube mesh.
- Color left and right hands differently.
- This approximates HaWoR's surface look, but it is not a real MANO mesh.

Pros:

- Works with current MediaPipe substitute backend.
- Useful before HaWoR/MANO is fully integrated.
- Good for pipeline debugging.

Cons:

- Fingers look like inflated joint hulls or tubes.
- No anatomically correct hand mesh.
- Does not match HaWoR page quality exactly.

#### Tier 2: HaWoR MANO Mesh Renderer

This is the target GPU-machine implementation.

Use:

- `verts_cam.npy`
- `verts_world.npy`
- `faces.npy`
- camera intrinsics from `manifest.json`
- camera poses from `trajectory.tum`

Camera-view panel:

1. Load the raw frame.
2. Render `verts_cam[t, hand]` with MANO `faces`.
3. Project mesh vertices with camera intrinsics.
4. Alpha-composite colored hand surfaces on top of the raw frame.
5. Optionally draw object/camera motion hints if available.

Top-view panel:

1. Use `verts_world[t]`.
2. Convert world coordinates into plot coordinates:
   - lateral: `X`
   - depth: `Z`
   - height: `-Y`
3. Render a gray floor plane.
4. Render colored MANO meshes.
5. Draw camera center and a short camera trajectory trail.
6. Draw a small frustum or direction arrow.

Side-view panel:

1. Use the same `verts_world[t]`.
2. View along lateral axis so depth and height are visible.
3. Render mesh shadows onto the floor for depth perception.
4. Draw camera/frustum marker.

### Recommended Renderer Implementation

Add a new module:

```text
src/aoe_pipeline/hawor_render.py
```

Core functions:

```python
def render_hawor_demo(
    clip_dir: Path,
    out_mp4: Path,
    fps: float = 30.0,
    size: int = 720,
    prefer_mesh: bool = True,
) -> Path:
    ...

def load_hawor_meshes(clip_dir: Path) -> MeshBundle | None:
    ...

def render_camera_panel(frame, verts_cam, faces, intrinsics) -> np.ndarray:
    ...

def render_top_panel(verts_world, faces, camera_pose, history) -> np.ndarray:
    ...

def render_side_panel(verts_world, faces, camera_pose, history) -> np.ndarray:
    ...
```

Add a CLI command:

```python
@app.command("render-hawor-demo")
def render_hawor_demo_cmd(
    clip_dir: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(None),
    fps: float = typer.Option(30.0),
    size: int = typer.Option(720),
    prefer_mesh: bool = typer.Option(True),
) -> None:
    ...
```

### Rendering Library Choice

Preferred on GPU machine:

- `pyrender` or `trimesh` + OpenGL/EGL for real mesh rendering.
- Fallback: Matplotlib `Poly3DCollection` for CPU-only rendering.

Recommended dependency split:

```toml
[project.optional-dependencies]
render = ["trimesh>=4.0", "pyrender>=0.1", "imageio[ffmpeg]>=2.34"]
```

If EGL/OpenGL is painful on the GPU server, start with Matplotlib. It is slower
but easier to make deterministic and headless.

### Visual Style Requirements

Match the HaWoR demo style:

- output video should be square, default `720x720`
- 2x2 panels with minimal spacing
- white panel labels:
  - `Input video`
  - `Camera view`
  - `Top view`
  - `Side view`
- left hand color: pink/purple
- right hand color: cyan/blue
- world floor: matte gray
- camera/frustum marker: small purple wireframe
- top/side view should include a short trajectory tail
- shadows are optional but strongly recommended for depth perception

### First Implementation Milestone

Build a working renderer using current AoE outputs:

```bash
aoe-pipeline render-hawor-demo \
  --clip-dir output/<clip> \
  --out output/<clip>/viz/hawor_demo_joint_hull.mp4 \
  --prefer-mesh false
```

Acceptance criteria:

- MP4 is created.
- Resolution is 720x720.
- Four panels are visible.
- Raw input panel matches source video.
- Camera-view panel shows projected colored hands.
- Top/side panels show 3D hand geometry and camera marker.

### Final GPU Implementation Milestone

After HaWoR wrapper exports MANO meshes:

```bash
aoe-pipeline render-hawor-demo \
  --clip-dir output_hybrid/<clip> \
  --out output_hybrid/<clip>/viz/hawor_demo.mp4 \
  --fps 30 \
  --size 720 \
  --prefer-mesh true

aoe-pipeline check-hawor-demo \
  --clip-dir output_hybrid/<clip> \
  --demo-mp4 output_hybrid/<clip>/viz/hawor_demo.mp4 \
  --require-mesh true \
  --expected-size 720
```

Acceptance criteria:

- Renderer uses `verts_cam.npy`, `verts_world.npy`, and `faces.npy`.
- Camera-view hands align with the RGB hands.
- Top/side view shows smooth MANO surfaces, not 21-joint hulls.
- Left/right hands keep stable colors and do not swap.
- Video resembles HaWoR's page demo layout and pacing.

### HaWoR Wrapper Extension For Rendering

Extend the wrapper contract from the hand reconstruction section. In addition
to joints, write:

```text
<external_dir>/verts_cam.npy    # (T,2,V,3), NaN for missing hands
<external_dir>/verts_world.npy  # (T,2,V,3), NaN for missing hands
<external_dir>/faces.npy        # (F,3), int triangle indices
```

Then update AoE's hybrid importer to copy these files into:

```text
<clip_dir>/hands/verts_cam.npy
<clip_dir>/hands/verts_world.npy
<clip_dir>/hands/faces.npy
```

If HaWoR only exports MANO parameters, convert them to vertices inside the
wrapper, not inside AoE. The wrapper already owns the HaWoR/MANO environment and
therefore has the correct MANO layer and model files.

### Testing Plan

Unit tests without GPU:

1. Create fake `verts_cam.npy`, `verts_world.npy`, and `faces.npy`.
2. Render 3 frames.
3. Assert output MP4 exists and has expected dimensions.
4. Delete mesh files and assert the renderer falls back to joint-hull mode.

Manual visual tests:

1. Compare one output frame against the HaWoR page structure.
2. Verify hand colors and panel labels.
3. Verify top/side panels move coherently with the source clip.
4. Verify camera-view overlay is not mirrored.

Useful inspection command:

```bash
python - <<'PY'
import cv2
p = "output_hybrid/<clip>/viz/hawor_demo.mp4"
cap = cv2.VideoCapture(p)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
PY
```

## Failure Modes And Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| HaWoR output frame count differs from AoE frame count | wrapper processed a different FPS/stride | use AoE `frames_dir` as source of truth |
| Left/right hands swapped | handedness convention mismatch | map HaWoR output to AoE slot convention: 0=Left, 1=Right |
| Many NaNs in HaWoR output | bad crop boxes or detector misses | use MediaPipe bbox expansion and fallback |
| World hands drift far away | trajectory coordinate convention mismatch | verify `trajectory.tum` is camera-to-world and units are consistent |
| MANO files missing | MANO license files not installed | download from MANO site and configure HaWoR path |
| CUDA/PyTorch errors | HaWoR environment mismatch | rebuild the shared uv env or Docker image with the pinned CUDA/PyTorch stack |

## Alternative Plans

### Alternative A: HaWoR Only

Use `hands.backend: original`.

Best when:

- HaWoR already handles detection/cropping well for your videos.
- You want fewer moving parts.

Risk:

- More brittle on frames where HaWoR misses hands.
- Harder to debug failed detections.

### Alternative B: MediaPipe Only

Use the current default substitute backend.

Best when:

- You need fast local iteration.
- No GPU or MANO files are available.

Risk:

- No MANO mesh.
- Lower 3D/world-space fidelity.

### Alternative C: HaMeR Instead Of HaWoR

HaMeR can be easier to run in some environments and also uses MANO.

Best when:

- HaWoR setup is blocked.
- You only need strong per-frame MANO reconstruction and can accept weaker
  world-space temporal modeling.

Risk:

- Less aligned with the AoE paper than HaWoR.
- Additional work needed to integrate camera/world trajectories.

## Definition Of Done

The GPU migration is complete when:

1. `configs/hawor_hybrid_gpu.yaml` exists and points to the real wrapper.
2. `aoe-pipeline run --config configs/hawor_hybrid_gpu.yaml` completes.
3. Final outputs exist:
   - `hands/joints_world.npy`
   - `hands/joints_cam.npy`
   - `hands/joints_2d.npy`
   - `hands/verts_cam.npy`
   - `hands/verts_world.npy`
   - `hands/faces.npy`
   - `hands/meta.json`
4. `hands/meta.json` reports `backend: hybrid`.
5. QC runs on the imported hybrid outputs.
6. `aoe-pipeline render-hawor-demo --prefer-mesh true` writes a 720x720 MP4.
7. `aoe-pipeline check-hawor-demo --require-mesh true` passes on the rendered
   clip.
8. Visualization overlays are coherent and the rendered four-panel video uses
   MANO surfaces, not the joint-hull fallback.
9. Tests pass:

```bash
pytest -q
```
