"""Homography / Bird's-Eye-View transformation."""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import cv2
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Tuple[float, float]]]


class PerspectiveTransformer:
    """Projects image-plane points onto a top-down 2D plane."""

    def __init__(self, src_pts: ArrayLike, dst_pts: ArrayLike) -> None:
        """Compute the homography matrix.

        Args:
            src_pts: 4 ROI corners in the camera frame (TL, TR, BR, BL).
            dst_pts: 4 matching corners on the top-down plane.

        Raises:
            ValueError: If either array is not of shape ``(4, 2)``.
        """
        self.src_pts = np.asarray(src_pts, dtype=np.float32)
        self.dst_pts = np.asarray(dst_pts, dtype=np.float32)
        if self.src_pts.shape != (4, 2) or self.dst_pts.shape != (4, 2):
            raise ValueError("src_pts and dst_pts must both have shape (4, 2).")
        self.matrix: np.ndarray = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self._roi_contour = self.src_pts.reshape(-1, 1, 2)

    def transform_points(self, points: ArrayLike) -> np.ndarray:
        """Project points to the bird's-eye plane.

        Args:
            points: Iterable of ``(x, y)`` feet positions.

        Returns:
            ``(N, 2)`` float32 array (``(0, 2)`` if the input is empty).
        """
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.matrix)
        return out.reshape(-1, 2)

    def inside_roi(self, points: ArrayLike) -> np.ndarray:
        """Boolean mask: which points fall inside (or on) the source ROI polygon."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return np.array(
            [cv2.pointPolygonTest(self._roi_contour, (float(x), float(y)), False) >= 0 for x, y in pts],
            dtype=bool,
        )
