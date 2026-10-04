"""Misc helpers: FPS counter, source parsing and interactive ROI selection."""
from __future__ import annotations

import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class FPSCounter:
    """Exponentially smoothed frames-per-second counter."""

    def __init__(self, smoothing: float = 0.9) -> None:
        """Args: smoothing: weight of the previous estimate, in ``[0, 1)``."""
        self.smoothing = smoothing
        self._prev = time.perf_counter()
        self._fps = 0.0

    def tick(self) -> float:
        """Register a processed frame and return the smoothed FPS."""
        now = time.perf_counter()
        dt = now - self._prev
        self._prev = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0.0 else (
                self.smoothing * self._fps + (1.0 - self.smoothing) * inst
            )
        return self._fps


def parse_source(source: str) -> Union[int, str]:
    """Convert ``"0"`` to webcam index ``0``; keep file paths/URLs as strings."""
    return int(source) if source.isdigit() else source


def select_points(
    frame: np.ndarray,
    num_points: int = 4,
    window_name: str = "Select ROI",
) -> Optional[np.ndarray]:
    """Let the user click ``num_points`` ROI corners on a frame.

    Controls: left-click add point, right-click undo, ``r`` reset,
    ``Enter``/``Space`` confirm (after all points), ``q``/``Esc`` cancel.
    Click order: top-left, top-right, bottom-right, bottom-left.

    Returns:
        ``(num_points, 2)`` float32 array, or ``None`` if cancelled.
    """
    points: List[Tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    result: Optional[np.ndarray] = None
    while True:
        canvas = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(canvas, p, 6, (0, 255, 255), -1)
            cv2.putText(canvas, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        if len(points) > 1:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)],
                          len(points) == num_points, (0, 255, 255), 2)

        if len(points) < num_points:
            msg = f"Click point {len(points) + 1}/{num_points} (TL, TR, BR, BL)"
        else:
            msg = "Enter/Space = confirm | R = reset"
        cv2.putText(canvas, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Right-click = undo | Q/Esc = cancel", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32) and len(points) == num_points:
            result = np.array(points, dtype=np.float32)
            break
        if key == ord("r"):
            points.clear()
        if key in (ord("q"), 27):
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)
    return result
