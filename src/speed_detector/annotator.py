"""Draw boxes, IDs, trails, and speed labels on frames."""

from __future__ import annotations

from typing import cast

import numpy as np

from speed_detector.types import SpeedReading, Track


class Annotator:
    """Render the pipeline's output onto frames using ``supervision`` annotators."""

    def __init__(self, trace_length: int = 30) -> None:
        import supervision as sv

        self._sv = sv
        self._box = sv.BoxAnnotator()
        self._label = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_LEFT)
        self._trace = sv.TraceAnnotator(trace_length=trace_length)

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        speeds: dict[int, SpeedReading],
    ) -> np.ndarray:
        scene: np.ndarray = frame.copy()
        if not tracks:
            return scene

        xyxy = np.array(
            [[t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2] for t in tracks], dtype=np.float32
        ).reshape(-1, 4)
        confidence = np.array([t.confidence for t in tracks], dtype=np.float32)
        class_id = np.array([int(t.cls) for t in tracks], dtype=np.int64)
        tracker_id = np.array([t.id for t in tracks], dtype=np.int64)

        dets = self._sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id, tracker_id=tracker_id
        )
        scene = self._trace.annotate(scene=scene, detections=dets)
        scene = self._box.annotate(scene=scene, detections=dets)
        labels = [self._label_for(t, speeds) for t in tracks]
        return cast(np.ndarray, self._label.annotate(scene=scene, detections=dets, labels=labels))

    @staticmethod
    def _label_for(track: Track, speeds: dict[int, SpeedReading]) -> str:
        reading = speeds.get(track.id)
        kind = track.cls.name.lower()
        if reading is not None:
            return f"#{track.id} {kind} {reading.speed_mph:.0f} mph"
        return f"#{track.id} {kind}"
