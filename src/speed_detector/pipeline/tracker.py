"""Multi-object tracking via ByteTrack (``trackers`` package).

``supervision``'s bundled ``ByteTrack`` was deprecated in 0.28 and is removed in
0.31; the algorithm now lives in the standalone ``trackers`` package, whose
``update`` method replaces ``update_with_detections``. Swapping to BoT-SORT is
a one-line change (``BotsortTracker``).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from speed_detector.types import BBox, Detection, Track, VehicleClass


class Tracker(Protocol):
    """Associates detections across frames into persistent track identities."""

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        """Return the tracked vehicles for this frame."""
        ...


class ByteTrackTracker:
    """ByteTrack via the ``trackers`` package — fast and CPU-friendly."""

    def __init__(self, frame_rate: float = 30.0) -> None:
        import supervision as sv
        from trackers import ByteTrackTracker as _ByteTrack

        self._sv = sv
        self._tracker = _ByteTrack(frame_rate=frame_rate)

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        xyxy = np.array(
            [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
            dtype=np.float32,
        ).reshape(-1, 4)
        confidence = (
            np.array([d.confidence for d in detections], dtype=np.float32)
            if detections
            else np.array([], dtype=np.float32)
        )
        class_id = (
            np.array([int(d.cls) for d in detections], dtype=np.int64)
            if detections
            else np.array([], dtype=np.int64)
        )
        sv_dets = self._sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._tracker.update(sv_dets)

        out: list[Track] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i].tolist()
            out.append(
                Track(
                    id=int(tracked.tracker_id[i]),
                    bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    confidence=float(tracked.confidence[i]),
                    cls=VehicleClass(int(tracked.class_id[i])),
                    frame_index=frame_index,
                )
            )
        return out
