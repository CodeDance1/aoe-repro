#!/usr/bin/env python3
"""Runtime GPU/import check for the HaWoR Docker image or shared uv env."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def main() -> int:
    import torch

    required = [
        "torch",
        "torchvision",
        "cv2",
        "mediapipe",
        "pyrender",
        "smplx",
        "mmcv",
        "torch_scatter",
        "pytorch3d",
        "aoe_pipeline",
    ]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"missing imports: {missing}")

    print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is false; run container with --gpus all")
    print("cuda device:", torch.cuda.get_device_name(0))
    a = torch.randn((1024, 1024), device="cuda")
    b = torch.randn((1024, 1024), device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("cuda matmul ok:", tuple(c.shape), str(c.dtype))

    hawor_root = Path(os.environ.get("HAWOR_ROOT", "/opt/aoe-repo/external/HaWoR"))
    required_files = [
        hawor_root / "_DATA/data/mano/MANO_RIGHT.pkl",
        hawor_root / "_DATA/data_left/mano_left/MANO_LEFT.pkl",
        hawor_root / "weights/external/detector.pt",
        hawor_root / "weights/external/droid.pth",
        hawor_root / "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth",
        hawor_root / "weights/hawor/checkpoints/hawor.ckpt",
        hawor_root / "weights/hawor/checkpoints/infiller.pt",
        hawor_root / "weights/hawor/model_config.yaml",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise SystemExit("missing MANO/model files:\n" + "\n".join(missing_files))
    print("HaWoR MANO/model files ok:", hawor_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
