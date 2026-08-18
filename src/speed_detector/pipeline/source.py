"""Video source abstraction — file, RTSP, USB webcam.

Phase 2 adds :class:`RTSPSource` and :class:`WebcamSource` alongside the
:class:`FileSource` from Phase 1, plus an :func:`open_source` factory that
picks the right implementation from a user-supplied string. YouTube ingest
(via ``yt-dlp``) is deferred to Phase 4; the threaded drop-on-overflow buffer
described in ``ARCHITECTURE.md §5.1`` also arrives in Phase 4.
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

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames until EOF."""
        ...


class _CvSource:
    """Common ``cv2.VideoCapture`` plumbing for file / RTSP / USB sources."""

    def __init__(
        self,
        src: str | int,
        *,
        buffer_size: int = 1,
        is_file: bool = False,
    ) -> None:
        if is_file and not Path(str(src)).is_file():
            raise FileNotFoundError(f"Video file not found: {src}")
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {src}")
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


class FileSource(_CvSource):
    """Read frames sequentially from a video file."""

    def __init__(self, path: str, buffer_size: int = 2) -> None:
        super().__init__(path, buffer_size=buffer_size, is_file=True)


class RTSPSource(_CvSource):
    """Read frames from an RTSP/RTMP/HTTP stream."""

    def __init__(self, url: str, buffer_size: int = 1) -> None:
        super().__init__(url, buffer_size=buffer_size, is_file=False)


class WebcamSource(_CvSource):
    """Read frames from a local USB webcam."""

    def __init__(self, index: int = 0, buffer_size: int = 1) -> None:
        super().__init__(index, buffer_size=buffer_size, is_file=False)


def open_source(spec: str) -> VideoSource:
    """Pick the right source implementation from a user-supplied string."""
    lowered = spec.lower()
    if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return RTSPSource(spec)
    if spec.isdigit():
        return WebcamSource(int(spec))
    return FileSource(spec)
