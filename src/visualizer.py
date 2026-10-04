"""OpenCV drawing utilities."""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

import config

Color = Tuple[int, int, int]


class Visualizer:
    """Draws detections, violations, HUD and a bird's-eye inset on frames."""

    def __init__(
        self,
        color_safe: Color = config.COLOR_SAFE,
        color_violation: Color = config.COLOR_VIOLATION,
    ) -> None:
        self.color_safe = color_safe
        self.color_violation = color_violation
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # ------------------------------------------------------------------ #
    # Main overlay
    # ------------------------------------------------------------------ #
    def draw_detections(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray,
        bottom_points: np.ndarray,
        violating_indices: Set[int],
        violating_pairs: List[Tuple[int, int]],
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Draw boxes, feet anchors, violation lines and the HUD (in place).

        Args:
            frame: BGR frame to draw on.
            bboxes: ``(N, 4)`` boxes ``[x1, y1, x2, y2]``.
            bottom_points: ``(N, 2)`` feet positions.
            violating_indices: Indices of people in violation.
            violating_pairs: ``(i, j)`` pairs in violation.
            fps: Optional FPS value shown in the HUD.

        Returns:
            The same ``frame`` for convenience.
        """
        total = len(bboxes)

        for pi, pj in violating_pairs:
            cv2.line(frame, tuple(map(int, bottom_points[pi])),
                     tuple(map(int, bottom_points[pj])), self.color_violation, 2)

        for idx in range(total):
            x1, y1, x2, y2 = map(int, bboxes[idx])
            color = self.color_violation if idx in violating_indices else self.color_safe
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, tuple(map(int, bottom_points[idx])), 5, color, -1)
            cv2.circle(frame, tuple(map(int, bottom_points[idx])), 5, (255, 255, 255), 1)

        self._draw_hud(frame, total, len(violating_indices), fps)
        return frame

    def _draw_hud(self, frame: np.ndarray, total: int, violations: int, fps: Optional[float]) -> None:
        """Draw the semi-transparent stats panel and an alert banner."""
        safe = total - violations
        lines = [
            (f"Total Persons: {total}", (255, 255, 255)),
            (f"Violations: {violations}", self.color_violation if violations else (255, 255, 255)),
            (f"Safe: {safe}", self.color_safe),
        ]
        if fps is not None:
            lines.append((f"FPS: {fps:.1f}", (255, 255, 0)))

        line_h, panel_w = 28, 230
        panel_h = 14 + line_h * len(lines)
        h, w = frame.shape[:2]
        x0, y0 = 10, 10
        x1, y1 = min(x0 + panel_w, w), min(y0 + panel_h, h)

        roi = frame[y0:y1, x0:x1]  # darken only the panel area (cheap)
        roi[:] = (roi * 0.4).astype(np.uint8)

        for k, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (x0 + 10, y0 + 28 + k * line_h),
                        self.font, 0.7, color, 2, cv2.LINE_AA)

        if violations > 0:
            msg = "SOCIAL DISTANCING ALERT"
            (tw, th), _ = cv2.getTextSize(msg, self.font, 0.9, 2)
            bx = (w - tw) // 2
            cv2.rectangle(frame, (bx - 10, 10), (bx + tw + 10, 20 + th + 10),
                          self.color_violation, -1)
            cv2.putText(frame, msg, (bx, 20 + th), self.font, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Bird's-eye inset
    # ------------------------------------------------------------------ #
    def draw_birdseye(
        self,
        frame: np.ndarray,
        transformed_points: np.ndarray,
        violating_indices: Set[int],
        violating_pairs: List[Tuple[int, int]],
        bev_size: Tuple[int, int],
        threshold_px: Optional[float] = None,
        height_ratio: float = 0.35,
    ) -> np.ndarray:
        """Render a small top-down map in the bottom-right corner (in place).

        When ``threshold_px`` is given, each person gets a circle of radius
        ``threshold_px / 2``; overlapping circles mean a violation.
        """
        bev_w, bev_h = bev_size
        canvas = np.full((bev_h, bev_w, 3), 35, dtype=np.uint8)

        for gx in range(0, bev_w, 100):
            cv2.line(canvas, (gx, 0), (gx, bev_h), (60, 60, 60), 1)
        for gy in range(0, bev_h, 100):
            cv2.line(canvas, (0, gy), (bev_w, gy), (60, 60, 60), 1)

        pts = np.round(np.asarray(transformed_points).reshape(-1, 2)).astype(int)

        for pi, pj in violating_pairs:
            cv2.line(canvas, tuple(pts[pi]), tuple(pts[pj]), self.color_violation, 2)
        for idx, (x, y) in enumerate(pts):
            color = self.color_violation if idx in violating_indices else self.color_safe
            if threshold_px:
                cv2.circle(canvas, (int(x), int(y)), int(threshold_px / 2), color, 1)
            cv2.circle(canvas, (int(x), int(y)), 7, color, -1)
        cv2.rectangle(canvas, (0, 0), (bev_w - 1, bev_h - 1), (200, 200, 200), 2)

        fh, fw = frame.shape[:2]
        inset_h = int(fh * height_ratio)
        inset_w = int(bev_w * inset_h / bev_h)
        margin = 10
        if inset_h < 20 or inset_w + margin > fw or inset_h + margin > fh:
            return frame  # frame too small for an inset

        inset = cv2.resize(canvas, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
        y0, x0 = fh - inset_h - margin, fw - inset_w - margin
        frame[y0:y0 + inset_h, x0:x0 + inset_w] = inset
        cv2.putText(frame, "Bird's-eye", (x0 + 4, y0 + 16), self.font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return frame
