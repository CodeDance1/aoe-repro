#!/usr/bin/env python3
"""Pre-flight self-check for the HaWoR faithful stage on a GPU box.

Verifies the GPU/CUDA toolchain, the HaWoR checkout + checkpoints, the MANO files,
the conda env, and (with --check-imports) that HaWoR's inference imports resolve —
the "step 0" items the `hands_hawor` stage needs. Run before
``aoe-pipeline run --config configs/faithful.yaml``.

    python scripts/check_hawor_env.py --hawor-dir third_party/HaWoR --check-imports

Exits non-zero if any *required* check fails. Safe to run anywhere (no GPU needed
to run the checker itself — it just reports what's missing).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"

CKPTS = {
    "weights/external/droid.pth": "DROID-SLAM",
    "weights/external/detector.pt": "hand detector",
    "weights/hawor/checkpoints/hawor.ckpt": "HaWoR",
    "weights/hawor/checkpoints/infiller.pt": "infiller",
    "weights/hawor/model_config.yaml": "model config",
    "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth": "Metric3D",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor-dir", default="third_party/HaWoR")
    ap.add_argument("--mano-dir", default=None, help="default <hawor-dir>/_DATA/data/mano")
    ap.add_argument("--conda-env", default="hawor")
    ap.add_argument("--check-imports", action="store_true",
                    help="probe HaWoR imports inside the conda env (needs the env built)")
    args = ap.parse_args()

    hawor = Path(args.hawor_dir)
    mano = Path(args.mano_dir) if args.mano_dir else hawor / "_DATA/data/mano"
    conda = shutil.which("conda") or shutil.which("mamba")
    results: list[tuple[bool, bool, str, str]] = []  # (required, ok, label, detail)

    def chk(required, ok, label, detail=""):
        results.append((required, ok, label, detail))

    # --- system toolchain ---
    smi = shutil.which("nvidia-smi")
    chk(True, bool(smi), "nvidia-smi (GPU/driver)", _gpu() if smi else "no NVIDIA GPU detected")
    nvcc = shutil.which("nvcc")
    chk(True, bool(nvcc), "nvcc (CUDA toolkit)",
        _nvcc() if nvcc else "needed to compile DROID-SLAM CUDA ext (use the AWS DLAMI)")
    chk(True, bool(conda), "conda/mamba", "" if conda else "install miniconda")
    chk(False, bool(shutil.which("ffmpeg")), "ffmpeg", "")

    # --- HaWoR repo + checkpoints ---
    chk(True, hawor.exists(), "HaWoR repo", str(hawor.resolve()) if hawor.exists() else "run envs/hawor.sh")
    for rel, what in CKPTS.items():
        p = hawor / rel
        chk(True, p.exists(), f"ckpt: {what}", "" if p.exists() else f"missing {p}")

    # --- MANO (license-gated) ---
    for f in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
        ok = (mano / f).exists()
        chk(True, ok, f"MANO: {f}", "" if ok else f"register + place under {mano}")

    # --- conda env ---
    env_ok = bool(conda) and _env_exists(conda, args.conda_env)
    chk(True, env_ok, f"conda env '{args.conda_env}'", "" if env_ok else "create via envs/hawor.sh")

    # --- optional: HaWoR import probe (step 0) ---
    if args.check_imports:
        if env_ok and hawor.exists():
            ok, msg = _import_probe(conda, args.conda_env, hawor)
            chk(True, ok, "HaWoR imports resolve (step 0)", msg)
        else:
            chk(False, False, "HaWoR imports (skipped)", "needs the env + repo first")

    # --- report ---
    print()
    blockers = 0
    for required, ok, label, detail in results:
        mark = OK if ok else (BAD if required else WARN)
        blockers += 1 if (required and not ok) else 0
        print(f"  {mark} {label}" + (f"  — {detail}" if detail else ""))
    print()
    if blockers:
        print(f"{BAD} {blockers} required check(s) failed — fix the above before running the faithful pipeline.")
        return 1
    print(f"{OK} all required checks passed — ready: aoe-pipeline run --config configs/faithful.yaml")
    return 0


def _gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else ""
    except Exception:
        return ""


def _nvcc() -> str:
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        return next((ln.strip() for ln in out.stdout.splitlines() if "release" in ln), "")
    except Exception:
        return ""


def _env_exists(conda: str, name: str) -> bool:
    try:
        out = subprocess.run([conda, "env", "list"], capture_output=True, text=True, timeout=30)
        return any(ln.split() and ln.split()[0] == name for ln in out.stdout.splitlines())
    except Exception:
        return False


def _import_probe(conda: str, env: str, hawor: Path) -> tuple[bool, str]:
    code = (
        "from lib.eval_utils.custom_utils import run_mano, run_mano_left;"
        "from scripts.scripts_test_video.detect_track_video import detect_track_video;"
        "from scripts.scripts_test_video.hawor_video import "
        "hawor_infiller, hawor_motion_estimation, hawor_slam;"
        "print('imports-ok')"
    )
    try:
        out = subprocess.run([conda, "run", "--no-capture-output", "-n", env, "python", "-c", code],
                             capture_output=True, text=True, timeout=600, cwd=str(hawor.resolve()))
        if "imports-ok" in out.stdout:
            return True, "run_mano + pipeline fns import (verify joint_map on first run)"
        tail = (out.stderr or out.stdout).strip().splitlines()[-2:]
        return False, "import failed (adjust paths in hawor_export.py): " + " ".join(tail)
    except Exception as exc:
        return False, f"probe error: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
