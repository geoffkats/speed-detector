"""Video source abstraction — RTSP, file, USB, YouTube.

Phase 1 implements :class:`FileSource` as the hello-world ingestion path.
``RTSPSource``, ``WebcamSource``, and ``YouTubeSource`` arrive in Phase 2.

The threaded drop-on-overflow buffer described in ``ARCHITECTURE.md §5.1``
arrives in Phase 4; ``FileSource`` reads sequentially for now, which is enough
to prove the ingestion path works.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class VideoSource(Protocol):
    """A source of timestamped video frames."""

    fps: float

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return ``(width, height)`` in pixels."""
        ...

    def next_frame(self) -> np.ndarray | None:
        """Return the next BGR frame, or ``None`` on EOF/error."""
        ...

    def close(self) -> None:
        """Release the underlying capture."""
        ...


class FileSource:
    """Read frames sequentially from a video file."""

    def __init__(self, path: str, buffer_size: int = 2) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)

    def next_frame(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return frame

    def close(self) -> None:
        self._cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        while (frame := self.next_frame()) is not None:
            yield frame
        self.close()
