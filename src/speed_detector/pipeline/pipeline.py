"""Pipeline orchestrator: source -> detect -> track -> calibrate -> speed -> annotate.

The core is a plain generator with no web, DB, or LLM dependency — every
downstream consumer (CLI, future API/analytics) reads the typed
:class:`FrameResult` it yields. See ``ARCHITECTURE.md §5.6``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from speed_detector.annotator import Annotator
from speed_detector.pipeline.calibrator import Calibrator
from speed_detector.pipeline.detector import Detector
from speed_detector.pipeline.source import VideoSource
from speed_detector.pipeline.speed import SpeedEstimator
from speed_detector.pipeline.tracker import Tracker
from speed_detector.types import SpeedReading, Track


@dataclass(frozen=True, slots=True)
class FrameResult:
    """The pipeline's output for one frame."""

    frame_index: int
    annotated: np.ndarray
    tracks: list[Track]
    speeds: dict[int, SpeedReading]


class Pipeline:
    """Run the full detection pipeline over a video source."""

    def __init__(
        self,
        source: VideoSource,
        detector: Detector,
        tracker: Tracker,
        calibrator: Calibrator,
        speed_estimator: SpeedEstimator,
        annotator: Annotator,
    ) -> None:
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._speed = speed_estimator
        self._annotator = annotator

    def run(self, max_frames: int | None = None) -> Iterator[FrameResult]:
        """Yield a :class:`FrameResult` per frame."""
        for frame_index, frame in enumerate(self._source):
            if max_frames is not None and frame_index >= max_frames:
                break
            detections = self._detector.detect(frame)
            tracks = self._tracker.update(detections, frame_index)
            speeds: dict[int, SpeedReading] = {}
            for track in tracks:
                reading = self._speed.update(track)
                if reading is not None:
                    speeds[track.id] = reading
            annotated = self._annotator.annotate(frame, tracks, speeds)
            yield FrameResult(
                frame_index=frame_index,
                annotated=annotated,
                tracks=tracks,
                speeds=speeds,
            )
