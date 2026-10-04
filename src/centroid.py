"""Reference-point selection for distance measurement.

Two methods are supported:

* ``centroid`` -- centre of the bounding box (baseline from the reference book).
* ``bottom``   -- bottom-centre of the bounding box, i.e. the feet position
  (our improvement: it lies on the ground plane, which is what the
  homography assumes).
"""
from __future__ import annotations

import numpy as np

METHODS = ("centroid", "bottom")


def get_reference_points(bboxes: np.ndarray, method: str = "centroid") -> np.ndarray:
    """Compute one reference point per bounding box.

    Args:
        bboxes: ``(N, 4)`` array of ``[x1, y1, x2, y2]``.
        method: ``"centroid"`` or ``"bottom"``.

    Returns:
        ``(N, 2)`` int32 array of ``(x, y)`` points (``(0, 2)`` if no boxes).

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")

    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)

    x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    y = (boxes[:, 1] + boxes[:, 3]) / 2.0 if method == "centroid" else boxes[:, 3]
    return np.stack([x, y], axis=1).astype(np.int32)
