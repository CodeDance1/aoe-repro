"""Joint-convention maps between MediaPipe-21, MANO-21, and dataset GT layouts.

MediaPipe hand landmark order (our prediction convention):
  0 wrist
  1-4   thumb  (CMC, MCP, IP, TIP)
  5-8   index  (MCP, PIP, DIP, TIP)
  9-12  middle (MCP, PIP, DIP, TIP)
  13-16 ring   (MCP, PIP, DIP, TIP)
  17-20 pinky  (MCP, PIP, DIP, TIP)

MANO 21-joint order groups all of one finger before the next, same as MediaPipe
for fingers but the canonical MANO finger order is index, middle, pinky, ring,
thumb. We expose explicit permutations and a generic ``remap`` helper. Verify
against a given dataset's documented layout before trusting absolute MPJPE.
"""

from __future__ import annotations

import numpy as np

MEDIAPIPE_LABELS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# MediaPipe index for each MANO joint slot (MANO finger order: index, middle,
# pinky, ring, thumb; wrist first). Adjust per dataset if its layout differs.
MEDIAPIPE_TO_MANO = [
    0,                  # wrist
    5, 6, 7, 8,         # index
    9, 10, 11, 12,      # middle
    17, 18, 19, 20,     # pinky
    13, 14, 15, 16,     # ring
    1, 2, 3, 4,         # thumb
]


# Inverse permutation: MANO-21 slot for each MediaPipe index (computed, not
# hand-derived, so the two maps can never drift apart).
MANO_TO_MEDIAPIPE: list[int] = np.argsort(MEDIAPIPE_TO_MANO).tolist()


def remap(joints: np.ndarray, order: list[int]) -> np.ndarray:
    """Reorder joints along the joint axis (second-to-last). ``joints``: (..., J, D)."""
    return np.asarray(joints)[..., order, :]


def to_mano(joints_mediapipe: np.ndarray) -> np.ndarray:
    return remap(joints_mediapipe, MEDIAPIPE_TO_MANO)


def from_mano(joints_mano: np.ndarray) -> np.ndarray:
    """MANO-21 order -> MediaPipe-21 order (the repo's storage convention)."""
    return remap(joints_mano, MANO_TO_MEDIAPIPE)
