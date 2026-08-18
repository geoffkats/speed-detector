"""Unit tests for the ground-plane speed estimator."""

from __future__ import annotations

import pytest

from speed_detector.pipeline.calibrator import Calibration, Calibrator
from speed_detector.pipeline.speed import SpeedEstimator
from speed_detector.types import BBox, Track, VehicleClass

RECT = Calibration(
    src=((0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)),
    real_width_m=5.0,
    real_length_m=10.0,
)


def _track(track_id: int, frame: int, cx: float, cy: float = 0.0) -> Track:
    """Build a 10x10 px box centred on (cx, cy)."""
    return Track(
        id=track_id,
        bbox=BBox(x1=cx - 5.0, y1=cy - 5.0, x2=cx + 5.0, y2=cy + 5.0),
        confidence=0.9,
        cls=VehicleClass.CAR,
        frame_index=frame,
    )


def test_uncalibrated_returns_none() -> None:
    estimator = SpeedEstimator(Calibrator(None), fps=10.0)
    assert estimator.update(_track(1, 0, 0.0)) is None


def test_insufficient_samples_returns_none() -> None:
    cal = Calibrator(RECT)
    estimator = SpeedEstimator(cal, fps=10.0, min_samples=4)
    assert estimator.update(_track(1, 0, 0.0)) is None
    assert estimator.update(_track(1, 1, 10.0)) is None
    assert estimator.update(_track(1, 2, 20.0)) is None
    # 4th sample crosses the min_samples threshold.
    assert estimator.update(_track(1, 3, 30.0)) is not None


def test_constant_velocity_yields_expected_speed() -> None:
    cal = Calibrator(RECT)
    estimator = SpeedEstimator(cal, fps=10.0)
    # 10 px/frame * (10 m / 100 px) = 1 m/frame; at 10 fps -> 10 m/s
    # -> 22.37 mph / 36.0 kph. EMA is exact for constant velocity.
    reading = None
    for frame in range(10):
        reading = estimator.update(_track(1, frame, frame * 10.0))
    assert reading is not None
    assert reading.speed_mph == pytest.approx(22.3694, abs=0.5)
    assert reading.speed_kph == pytest.approx(36.0, abs=0.5)
    assert reading.track_id == 1
    assert reading.cls == VehicleClass.CAR


def test_tracks_are_independent() -> None:
    cal = Calibrator(RECT)
    estimator = SpeedEstimator(cal, fps=10.0)
    for frame in range(12):
        estimator.update(_track(1, frame, frame * 10.0))  # 1 m/frame -> 10 m/s
        estimator.update(_track(2, frame, frame * 20.0))  # 2 m/frame -> 20 m/s
    fast = estimator.update(_track(2, 12, 240.0))
    slow = estimator.update(_track(1, 12, 120.0))
    assert fast is not None
    assert slow is not None
    assert fast.speed_mph > slow.speed_mph
