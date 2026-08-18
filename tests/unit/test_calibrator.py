"""Unit tests for the perspective-transform calibrator."""

from __future__ import annotations

import pytest

from speed_detector.pipeline.calibrator import Calibration, Calibrator

# A 100x50 px rectangle standing in for a 10m x 5m patch of road.
RECT = Calibration(
    src=((0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)),
    real_width_m=5.0,
    real_length_m=10.0,
)


def test_uncalibrated_raises() -> None:
    cal = Calibrator(None)
    assert not cal.is_calibrated
    with pytest.raises(RuntimeError):
        cal.to_world((1.0, 1.0))


def test_corners_map_to_real_world() -> None:
    cal = Calibrator(RECT)
    assert cal.is_calibrated
    assert cal.to_world((0.0, 0.0)) == pytest.approx((0.0, 0.0))
    assert cal.to_world((100.0, 0.0)) == pytest.approx((10.0, 0.0))
    assert cal.to_world((100.0, 50.0)) == pytest.approx((10.0, 5.0))
    assert cal.to_world((0.0, 50.0)) == pytest.approx((0.0, 5.0))


def test_midpoint_maps_linearly() -> None:
    cal = Calibrator(RECT)
    # Centre of the rectangle: (50, 25) px -> (5.0, 2.5) m.
    assert cal.to_world((50.0, 25.0)) == pytest.approx((5.0, 2.5))


def test_world_distance_is_euclidean() -> None:
    cal = Calibrator(RECT)
    assert cal.world_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert cal.world_distance((1.0, 1.0), (1.0, 1.0)) == pytest.approx(0.0)
