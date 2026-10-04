"""YOLO-based person detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO

import config


@dataclass
class Detections:
    """Container for per-frame detection results.

    Attributes:
        bboxes: ``(N, 4)`` int array of ``[x1, y1, x2, y2]``.
        bottom_points: ``(N, 2)`` int array of feet positions ``(x, y)``.
        confidences: ``(N,)`` float array of detection scores.
    """

    bboxes: np.ndarray
    bottom_points: np.ndarray
    confidences: np.ndarray

    @classmethod
    def empty(cls) -> "Detections":
        """Return a container with zero detections."""
        return cls(
            bboxes=np.empty((0, 4), dtype=np.int32),
            bottom_points=np.empty((0, 2), dtype=np.int32),
            confidences=np.empty((0,), dtype=np.float32),
        )

    def __len__(self) -> int:
        return int(self.bboxes.shape[0])

    def filter(self, mask: np.ndarray) -> "Detections":
        """Return a new ``Detections`` keeping only entries where ``mask`` is True."""
        mask = np.asarray(mask, dtype=bool)
        return Detections(self.bboxes[mask], self.bottom_points[mask], self.confidences[mask])


class PersonDetector:
    """Detects people in a frame using an Ultralytics YOLO model.

    Any detector exposing ``detect(frame) -> Detections`` can replace this
    class without touching the rest of the pipeline.
    """

    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        person_class_id: int = config.PERSON_CLASS_ID,
        device: Optional[str] = None,
    ) -> None:
        """Load the YOLO model.

        Args:
            model_path: Path/name of the weights (e.g. ``yolov8n.pt``).
            confidence_threshold: Minimum confidence to keep a detection.
            person_class_id: Class id of "person" (0 for COCO).
            device: Optional inference device (``"cpu"``, ``"cuda:0"``, ...).
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.device = device

    def detect(self, frame: np.ndarray) -> Detections:
        """Run inference on a BGR frame and return persons only.

        The bottom-center of each box ``(int((x1+x2)/2), int(y2))`` is used as
        the feet position (never the box centroid).

        Args:
            frame: BGR image of shape ``(H, W, 3)``.

        Returns:
            A :class:`Detections` instance (empty if nobody is found).
        """
        kwargs = {
            "classes": [self.person_class_id],
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(frame, **kwargs)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return Detections.empty()

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        keep = conf >= self.confidence_threshold
        xyxy, conf = xyxy[keep], conf[keep]
        if xyxy.shape[0] == 0:
            return Detections.empty()

        x1, y2, x2 = xyxy[:, 0], xyxy[:, 3], xyxy[:, 2]
        bottom = np.stack([(x1 + x2) / 2.0, y2], axis=1).astype(np.int32)
        return Detections(xyxy.astype(np.int32), bottom, conf.astype(np.float32))
