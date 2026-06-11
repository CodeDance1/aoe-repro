"""Tests for the faithful HaWoR stage: export glue, loader, joint maps, gating.

All GPU-free — they exercise the parsing/assembly/gating around the subprocess,
never HaWoR itself. ``scripts/hawor_export.py``'s pure helpers are loaded by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from aoe_pipeline.config import PipelineConfig
from aoe_pipeline.eval.joint_maps import MANO_TO_MEDIAPIPE, MEDIAPIPE_TO_MANO, from_mano, to_mano
from aoe_pipeline.schema import ClipManifest, MANOSequence
from aoe_pipeline.stages.base import ClipContext
from aoe_pipeline.stages.registry import available
from aoe_pipeline.stages.s4_hands_hawor import HandStageHaWoR, load_hawor_export

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hx = _load("hawor_export", "scripts/hawor_export.py")


# --- joint maps (still used by eval against MANO-order GT) ------------------
def test_mano_mediapipe_maps_are_inverse():
    j = np.random.default_rng(0).normal(size=(21, 3))
    assert np.allclose(from_mano(to_mano(j)), j)
    assert sorted(MANO_TO_MEDIAPIPE) == list(range(21))
    assert [MEDIAPIPE_TO_MANO[m] for m in MANO_TO_MEDIAPIPE] == list(range(21))


# --- wrapper pure glue (assemble_export / _project) ------------------------
def _raw(T: int = 4, valid=None) -> dict:
    rng = np.random.default_rng(0)
    joints = rng.normal(size=(2, T, 21, 3)).astype(np.float32) + np.array([0, 0, 2.0])  # in front
    return dict(
        joints_per_hand=joints,
        pred_trans=np.zeros((2, T, 3)), pred_rot=np.tile(np.eye(3), (2, T, 1, 1)),
        pred_hand_pose=np.zeros((2, T, 45)), pred_betas=np.zeros((2, T, 10)),
        pred_valid=np.ones((2, T), bool) if valid is None else valid,
        R_w2c=np.tile(np.eye(3), (T, 1, 1)), t_w2c=np.zeros((T, 3)),
        R_c2w=np.tile(np.eye(3), (T, 1, 1)), t_c2w=np.zeros((T, 3)),
        img_focal=500.0, width=640, height=480, fps=30.0,
    )


def test_assemble_export_shapes():
    ex = hx.assemble_export(**_raw(T=4))
    assert ex["joints_world"].shape == (4, 2, 21, 3)
    assert ex["joints_2d"].shape == (4, 2, 21, 2)
    assert ex["valid"].shape == (2, 4)
    assert len(ex["tum_lines"]) == 4


def test_assemble_export_reconciles_frame_counts():
    raw = _raw(T=5)
    raw["R_c2w"], raw["t_c2w"] = raw["R_c2w"][:3], raw["t_c2w"][:3]  # shorter camera track
    ex = hx.assemble_export(**raw)
    assert ex["joints_world"].shape[0] == 3   # truncated to the common min
    assert len(ex["tum_lines"]) == 3
    assert ex["trans"].shape[1] == 3


def test_assemble_export_nan_masks_invalid_frames():
    valid = np.ones((2, 4), bool)
    valid[0, 1] = False  # left hand invalid at frame 1
    ex = hx.assemble_export(**_raw(T=4, valid=valid))
    assert np.isnan(ex["joints_world"][1, 0]).all()   # masked -> NaN
    assert np.isnan(ex["joints_2d"][1, 0]).all()
    assert np.isfinite(ex["joints_world"][1, 1]).all()  # right hand untouched


def test_project_nan_for_behind_camera():
    K = np.array([[500.0, 0, 0], [0, 500.0, 0], [0, 0, 1.0]])
    R, t = np.eye(3)[None], np.zeros((1, 3))
    jw = np.zeros((1, 1, 21, 3), np.float32)
    jw[0, 0, :, 2] = -1.0                       # behind the camera
    assert np.isnan(hx._project(jw, R, t, K)[0, 0]).all()
    jw[0, 0, :, 2] = 2.0                        # in front
    assert np.isfinite(hx._project(jw, R, t, K)[0, 0]).all()


# --- export loader (no remap: HaWoR is already MediaPipe order) ------------
def _write_fake_export(clip_dir: Path, T: int = 4):
    (clip_dir / "hands").mkdir(parents=True)
    joints = np.zeros((T, 2, 21, 3), np.float32)
    for k in range(21):
        joints[:, :, k, 0] = k  # encode joint index to detect any accidental reorder
    np.savez_compressed(
        clip_dir / "hands" / "hawor_raw.npz",
        trans=np.zeros((2, T, 3)), global_orient=np.tile(np.eye(3), (2, T, 1, 1)),
        hand_pose=np.zeros((2, T, 45)), betas=np.zeros((2, T, 10)),
        valid=np.ones((2, T), bool), joints_world=joints, joints_2d=joints[..., :2],
    )
    lines = [f"{t/30:.6f} {t*0.1:.6f} 0 0 0 0 0 1" for t in range(T)]
    (clip_dir / "trajectory.tum").write_text("\n".join(lines) + "\n")


def test_load_hawor_export_copies_verbatim_and_writes(tmp_path):
    _write_fake_export(tmp_path)
    out = load_hawor_export(tmp_path)

    jw = out["joints_world"]
    assert jw.shape == (4, 2, 21, 3)
    for k in range(21):  # stored verbatim (MediaPipe order) — NOT remapped/scrambled
        assert jw[0, 0, k, 0] == k

    for f in ("joints_world.npy", "joints_2d.npy", "mano.npz"):
        assert (tmp_path / "hands" / f).exists()
    mano = MANOSequence.load(tmp_path / "hands" / "mano.npz")
    assert mano.num_frames == 4 and mano.hand_pose.shape == (2, 4, 45)
    assert out["poses"][1][0, 3] == 0.1 and len(out["poses"]) == 4


# --- stage gating -----------------------------------------------------------
def _ctx(tmp_path: Path) -> ClipContext:
    return ClipContext(clip_id="c", clip_dir=tmp_path, video_path=Path("v.mp4"),
                       config=PipelineConfig.default(),
                       manifest=ClipManifest(clip_id="c", video_path="v.mp4"))


def test_hands_hawor_registered_but_not_in_default_pipeline():
    assert "hands_hawor" in available()
    assert "hands_hawor" not in PipelineConfig.default().pipeline  # lite stays default


def test_stage_skips_without_config(tmp_path):
    c = _ctx(tmp_path)
    HandStageHaWoR({}).run(c)
    st = c.manifest.stages["hands_hawor"]
    assert st.status == "skipped"
    assert "hawor_dir" in st.info["reason"]


def test_stage_skips_when_repo_missing(tmp_path):
    c = _ctx(tmp_path)
    HandStageHaWoR({"hawor_dir": str(tmp_path / "nope")}).run(c)
    assert c.manifest.stages["hands_hawor"].status == "skipped"


def test_stage_skips_when_weights_missing(tmp_path):
    (tmp_path / "hawor").mkdir()  # repo exists but checkpoints not downloaded
    c = _ctx(tmp_path)
    HandStageHaWoR({"hawor_dir": str(tmp_path / "hawor")}).run(c)
    st = c.manifest.stages["hands_hawor"]
    assert st.status == "skipped"
    assert "weight missing" in st.info["reason"]


def test_check_hawor_env_script_reports_and_exits_nonzero(tmp_path):
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_hawor_env.py"),
         "--hawor-dir", str(tmp_path / "absent")],
        capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "required check" in r.stdout
    assert "HaWoR repo" in r.stdout


def test_faithful_config_parses_and_orders_stages():
    cfg = PipelineConfig.from_yaml(REPO / "configs" / "faithful.yaml")
    assert "hands_hawor" in cfg.pipeline
    assert "trajectory" not in cfg.pipeline and "hands" not in cfg.pipeline  # superseded
    assert cfg.pipeline.index("hands_hawor") < cfg.pipeline.index("segment")
