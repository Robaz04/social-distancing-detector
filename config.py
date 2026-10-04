"""Centralized configuration and hyper-parameters.

All tunable values live here. They are exposed both as module-level constants
(easy to import) and through the :class:`Settings` dataclass, which can be
overridden at runtime by a JSON file (``--config``) or CLI flags.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Tuple, Union

import numpy as np

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
MODEL_PATH: str = "yolov8n.pt"
PERSON_CLASS_ID: int = 0  # COCO "person"
CONFIDENCE_THRESHOLD: float = 0.5

# --------------------------------------------------------------------------- #
# Distance logic (in Bird's-Eye-View pixels, NOT real-world meters)
# --------------------------------------------------------------------------- #
DISTANCE_THRESHOLD_PX: float = 100.0

# --------------------------------------------------------------------------- #
# Colors (BGR)
# --------------------------------------------------------------------------- #
COLOR_SAFE: Tuple[int, int, int] = (0, 255, 0)
COLOR_VIOLATION: Tuple[int, int, int] = (0, 0, 255)

# --------------------------------------------------------------------------- #
# Perspective transform
# --------------------------------------------------------------------------- #
# Default ROI on the camera frame (order: top-left, top-right, bottom-right,
# bottom-left). Tuned for a 1280x720 frame -- override via --select-points or
# --config for your own video.
SRC_POINTS: np.ndarray = np.array(
    [[400, 250], [880, 250], [1180, 700], [100, 700]], dtype=np.float32
)

# Size of the top-down (bird's-eye) plane.
BEV_WIDTH: int = 400
BEV_HEIGHT: int = 600
DST_POINTS: np.ndarray = np.array(
    [[0, 0], [BEV_WIDTH, 0], [BEV_WIDTH, BEV_HEIGHT], [0, BEV_HEIGHT]],
    dtype=np.float32,
)

# --------------------------------------------------------------------------- #
# Runtime / UI
# --------------------------------------------------------------------------- #
FILTER_OUTSIDE_ROI: bool = False  # drop people whose feet fall outside the ROI
SHOW_BIRDSEYE: bool = True        # small top-down inset window overlay
WINDOW_NAME: str = "Social Distancing Detector"


@dataclass
class Settings:
    """Mutable runtime settings, initialised from the module-level defaults."""

    model_path: str = MODEL_PATH
    person_class_id: int = PERSON_CLASS_ID
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    distance_threshold_px: float = DISTANCE_THRESHOLD_PX
    color_safe: Tuple[int, int, int] = COLOR_SAFE
    color_violation: Tuple[int, int, int] = COLOR_VIOLATION
    src_points: np.ndarray = field(default_factory=lambda: SRC_POINTS.copy())
    dst_points: np.ndarray = field(default_factory=lambda: DST_POINTS.copy())
    filter_outside_roi: bool = FILTER_OUTSIDE_ROI
    show_birdseye: bool = SHOW_BIRDSEYE
    window_name: str = WINDOW_NAME

    @property
    def bev_size(self) -> Tuple[int, int]:
        """(width, height) of the bird's-eye plane, derived from ``dst_points``."""
        w = int(np.ceil(self.dst_points[:, 0].max()))
        h = int(np.ceil(self.dst_points[:, 1].max()))
        return max(w, 1), max(h, 1)

    @property
    def uses_default_src_points(self) -> bool:
        """True if the ROI still equals the built-in default."""
        return bool(np.array_equal(self.src_points, SRC_POINTS))

    def validate(self) -> None:
        """Raise ``ValueError`` if any setting is inconsistent."""
        for name in ("src_points", "dst_points"):
            if np.asarray(getattr(self, name)).shape != (4, 2):
                raise ValueError(f"{name} must have shape (4, 2).")
        for name in ("color_safe", "color_violation"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"{name} must be a BGR triple.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1].")
        if self.distance_threshold_px <= 0:
            raise ValueError("distance_threshold_px must be > 0.")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Settings":
        """Build settings from defaults overridden by a JSON file.

        Keys are case-insensitive and must match field names, e.g.
        ``{"distance_threshold_px": 120, "src_points": [[0,0],[1,0],[1,1],[0,1]]}``.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError("Config JSON must be an object at the top level.")

        valid = {f.name for f in fields(cls)}
        settings = cls()
        for key, value in raw.items():
            name = key.lower()
            if name not in valid:
                raise ValueError(f"Unknown config key: {key!r}")
            if name.endswith("_points"):
                value = np.asarray(value, dtype=np.float32)
            elif name.startswith("color_"):
                value = tuple(int(c) for c in value)
            setattr(settings, name, value)
        settings.validate()
        return settings
