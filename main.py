"""Entry point: real-time social distancing detection with Bird's-Eye View."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

import cv2
import numpy as np

import config
from config import Settings
from src.detector import PersonDetector
from src.distance import DistanceAnalyzer
from src.perspective import PerspectiveTransformer
from src.visualizer import Visualizer
from utils.helper import FPSCounter, parse_source, select_points

log = logging.getLogger("social_distancing")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    p = argparse.ArgumentParser(description="Real-time social distancing detection (YOLO + BEV).")
    p.add_argument("--source", default="0", help="Video file path, stream URL, or webcam index (default: 0).")
    p.add_argument("--config", default=None, help="Optional JSON file overriding config.py values.")
    p.add_argument("--select-points", action="store_true", help="Click 4 ROI points on the first frame.")
    p.add_argument("--no-select", action="store_true", help="Never prompt; use configured SRC_POINTS.")
    p.add_argument("--model", default=None, help="Override YOLO weights path.")
    p.add_argument("--conf", type=float, default=None, help="Override confidence threshold.")
    p.add_argument("--threshold", type=float, default=None, help="Override distance threshold (BEV px).")
    p.add_argument("--roi-only", action="store_true", help="Only analyse people standing inside the ROI.")
    p.add_argument("--no-birdseye", action="store_true", help="Hide the bird's-eye inset.")
    return p


def load_settings(args: argparse.Namespace) -> Settings:
    """Merge defaults, optional JSON config and CLI overrides (CLI wins)."""
    settings = Settings.from_json(args.config) if args.config else Settings()
    if args.model is not None:
        settings.model_path = args.model
    if args.conf is not None:
        settings.confidence_threshold = args.conf
    if args.threshold is not None:
        settings.distance_threshold_px = args.threshold
    if args.roi_only:
        settings.filter_outside_roi = True
    if args.no_birdseye:
        settings.show_birdseye = False
    settings.validate()
    return settings


def run(settings: Settings, source: object, select: bool) -> int:
    """Run the processing loop. Returns a process exit code."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video source: %r", source)
        return 1

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            log.error("Could not read the first frame from %r", source)
            return 1

        # --- ROI selection -------------------------------------------------
        if select:
            chosen: Optional[np.ndarray] = select_points(frame, 4, "Select ROI")
            if chosen is None:
                log.info("ROI selection cancelled.")
                return 0
            settings.src_points = chosen
            log.info("ROI selected. Reuse it via --config:\n%s",
                     json.dumps({"src_points": chosen.astype(int).tolist()}))
        elif settings.uses_default_src_points:
            log.warning("Using default SRC_POINTS (tuned for 1280x720); results may be off.")

        # --- Pipeline components (loosely coupled) -------------------------
        detector = PersonDetector(settings.model_path, settings.confidence_threshold,
                                  settings.person_class_id)
        transformer = PerspectiveTransformer(settings.src_points, settings.dst_points)
        analyzer = DistanceAnalyzer()
        visualizer = Visualizer(settings.color_safe, settings.color_violation)
        fps_counter = FPSCounter()
        bev_size = settings.bev_size
        threshold = settings.distance_threshold_px

        # --- Main loop -----------------------------------------------------
        while ok and frame is not None:
            dets = detector.detect(frame)

            if settings.filter_outside_roi and len(dets) > 0:
                dets = dets.filter(transformer.inside_roi(dets.bottom_points))

            bev_points = transformer.transform_points(dets.bottom_points)
            violating_indices, violating_pairs = analyzer.evaluate_violations(bev_points, threshold)

            fps = fps_counter.tick()
            visualizer.draw_detections(frame, dets.bboxes, dets.bottom_points,
                                       violating_indices, violating_pairs, fps)
            if settings.show_birdseye:
                visualizer.draw_birdseye(frame, bev_points, violating_indices,
                                         violating_pairs, bev_size, threshold)

            cv2.imshow(settings.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            ok, frame = cap.read()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args()
    try:
        settings = load_settings(args)
    except (OSError, ValueError) as exc:
        log.error("Invalid configuration: %s", exc)
        return 2

    select = args.select_points or (not args.no_select and settings.uses_default_src_points)
    return run(settings, parse_source(args.source), select)


if __name__ == "__main__":
    sys.exit(main())
