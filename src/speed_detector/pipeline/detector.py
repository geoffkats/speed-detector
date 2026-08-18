"""Vehicle detection — Ultralytics frontend (loads ``.pt`` or ``.onnx``).

The Ultralytics ``YOLO`` class runs ``.pt`` weights via PyTorch (the dev path)
and ``.onnx`` weights via ONNX Runtime (the production path) — same code, a
different weights file. Both return the typed ``Detection`` records from
:mod:`speed_detector.types`, keeping the rest of the pipeline detector-agnostic.

A bare-``onnxruntime`` detector (manual letterbox + NMS, no Ultralytics) is a
planned optimisation for the Docker image; for now ONNX Runtime is exercised
through Ultralytics' loader so the production path still avoids PyTorch.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from speed_detector.types import BBox, Detection, VehicleClass

# COCO class id -> our vehicle enum. Non-vehicle classes are filtered out.
COCO_TO_VEHICLE: dict[int, VehicleClass] = {
    1: VehicleClass.BICYCLE,
    2: VehicleClass.CAR,
    3: VehicleClass.MOTORCYCLE,
    5: VehicleClass.BUS,
    7: VehicleClass.TRUCK,
}


@runtime_checkable
class Detector(Protocol):
    """A vehicle detector over a single BGR frame."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return the vehicle detections in ``frame``."""
        ...


class UltralyticsDetector:
    """YOLO via the Ultralytics package (loads ``.pt`` or ``.onnx`` weights)."""

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        confidence: float = 0.35,
        iou: float = 0.7,
        device: str = "cpu",
    ) -> None:
        # Local imports keep the module importable without torch/onnxruntime
        # (e.g. for the geometry-only unit tests).
        import supervision as sv
        from ultralytics import YOLO

        self._sv = sv
        self._model = YOLO(weights, task="detect")
        self._conf = confidence
        self._iou = iou
        self._device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(
            frame, conf=self._conf, iou=self._iou, verbose=False, device=self._device
        )
        dets = self._sv.Detections.from_ultralytics(results[0])
        mask = np.isin(dets.class_id, list(COCO_TO_VEHICLE))
        dets = dets[mask]

        out: list[Detection] = []
        for i in range(len(dets)):
            x1, y1, x2, y2 = dets.xyxy[i].tolist()
            out.append(
                Detection(
                    bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    confidence=float(dets.confidence[i]),
                    cls=COCO_TO_VEHICLE[int(dets.class_id[i])],
                )
            )
        return out
