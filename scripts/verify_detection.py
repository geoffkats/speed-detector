"""Verify the detector + tracker + annotator on a single real image.

A one-shot smoke test for the CV stack: load the model, run detection and
tracking on one frame, draw the result, and report the vehicle classes found.

Usage::

    uv run python -m scripts.verify_detection path/to/traffic.jpg [annotated_out.jpg]

Useful as a fast sanity check that the weights load and YOLO detects real
vehicles before running the full video pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from speed_detector.annotator import Annotator
from speed_detector.config import Settings
from speed_detector.pipeline.detector import UltralyticsDetector
from speed_detector.pipeline.tracker import ByteTrackTracker


def main(image: str, out: str | None = None) -> None:
    frame = cv2.imread(image)
    if frame is None:
        raise SystemExit(f"Could not read image: {image}")

    settings = Settings()
    weights = settings.onnx_model_path
    if not Path(weights).is_file():
        weights = "yolo11n.pt"

    detector = UltralyticsDetector(
        weights=weights, confidence=settings.yolo_confidence, iou=settings.yolo_iou
    )
    detections = detector.detect(frame)
    tracks = ByteTrackTracker(frame_rate=1.0).update(detections, 0)
    annotated = Annotator().annotate(frame, tracks, {})

    by_class: dict[str, int] = {}
    for d in detections:
        by_class[d.cls.name] = by_class.get(d.cls.name, 0) + 1

    print(f"Image: {image}  ({frame.shape[1]}x{frame.shape[0]})")
    print(f"Weights: {weights}")
    print(f"Detected {len(detections)} vehicles: {by_class or 'none'}")

    if out:
        cv2.imwrite(out, annotated)
        print(f"Annotated image written to: {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit("Usage: python -m scripts.verify_detection <image> [out]")
    main(args[0], args[1] if len(args) > 1 else None)
