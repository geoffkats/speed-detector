"""Ground-plane speed estimation with EMA smoothing.

Each tracked vehicle keeps a short history of world positions; velocity is the
ground-plane distance travelled over elapsed time, smoothed with an exponential
moving average to suppress per-frame jitter. Returns ``None`` until the track
has enough samples, and ``None`` entirely when no calibration is set — speeds
without real-world units would be misleading, so the UI simply omits them.
"""

from __future__ import annotations

from collections import deque

from speed_detector.pipeline.calibrator import Calibrator
from speed_detector.types import SpeedReading, Track

_MPS_TO_MPH = 2.23693629
_MPS_TO_KPH = 3.6


class SpeedEstimator:
    """Per-track speed in m/s, mph, kph, smoothed via exponential moving average."""

    def __init__(
        self,
        calibrator: Calibrator,
        fps: float,
        history: int = 15,
        alpha: float = 0.3,
        min_samples: int = 4,
    ) -> None:
        self._calibrator = calibrator
        self._fps = fps
        self._window = history
        self._alpha = alpha
        self._min_samples = min_samples
        self._positions: dict[int, deque[tuple[float, float, int]]] = {}
        self._smoothed: dict[int, float] = {}

    def update(self, track: Track) -> SpeedReading | None:
        """Record a track's position and return a smoothed speed, or ``None``."""
        if not self._calibrator.is_calibrated:
            return None

        cx = (track.bbox.x1 + track.bbox.x2) / 2.0
        cy = (track.bbox.y1 + track.bbox.y2) / 2.0
        world = self._calibrator.to_world((cx, cy))

        hist = self._positions.setdefault(track.id, deque(maxlen=self._window))
        hist.append((world[0], world[1], track.frame_index))

        if len(hist) < self._min_samples:
            return None

        first_x, first_y, first_frame = hist[0]
        last_x, last_y, last_frame = hist[-1]
        dt_frames = last_frame - first_frame
        if dt_frames <= 0:
            return None

        dt_seconds = dt_frames / self._fps
        dist = self._calibrator.world_distance((first_x, first_y), (last_x, last_y))
        mps = dist / dt_seconds

        prev = self._smoothed.get(track.id, mps)
        smoothed = self._alpha * mps + (1.0 - self._alpha) * prev
        self._smoothed[track.id] = smoothed

        return SpeedReading(
            track_id=track.id,
            cls=track.cls,
            speed_mph=smoothed * _MPS_TO_MPH,
            speed_kph=smoothed * _MPS_TO_KPH,
            world_position_m=world,
            frame_index=track.frame_index,
        )
