from __future__ import annotations

import sys

import numpy as np

from aoe_pipeline.external import ExternalPipelineError, render_command, run_external
from aoe_pipeline.stages.s3_trajectory import _read_tum_poses


def test_render_command_rejects_unknown_placeholder():
    try:
        render_command("python {missing}", {"known": "x"})
    except ExternalPipelineError as exc:
        assert "unknown placeholder" in str(exc)
    else:
        raise AssertionError("expected ExternalPipelineError")


def test_run_external_formats_list_command(tmp_path):
    marker = tmp_path / "marker.txt"
    run_external(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path(r'{marker}').write_text('ok')",
        ],
        {"marker": marker},
    )

    assert marker.read_text() == "ok"


def test_read_tum_poses(tmp_path):
    path = tmp_path / "trajectory.tum"
    path.write_text(
        "0.000000 1 2 3 0 0 0 1\n"
        "0.033333 4 5 6 0 0 0 1\n"
    )

    poses = _read_tum_poses(path)

    assert len(poses) == 2
    assert np.allclose(poses[0][:3, 3], [1, 2, 3])
    assert np.allclose(poses[1][:3, :3], np.eye(3))
