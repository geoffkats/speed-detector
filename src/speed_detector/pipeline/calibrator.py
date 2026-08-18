"""Perspective transform: map pixel coordinates to real-world meters.

A 4-point calibration — the pixel corners of a known rectangle on the road,
plus that rectangle's real-world width and length in meters — yields a
homography that converts any pixel position to ground-plane meters. Speeds are
then computed on the ground plane, giving real mph/kph instead of the legacy
pixel heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Calibration:
    """A 4-point perspective calibration for one camera.

    ``src`` are the pixel corners of a known road rectangle, clockwise from the
    near-left corner. ``real_length_m`` is the road length along the travel
    direction spanned by the rectangle; ``real_width_m`` the lane width.
    """

    src: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    real_width_m: float
    real_length_m: float


class Calibrator:
    """Apply a perspective transform to convert pixels to meters."""

    def __init__(self, calibration: Calibration | None = None) -> None:
        self._calibration = calibration
        self._homography: np.ndarray | None = None
        if calibration is not None:
            src = np.array(calibration.src, dtype=np.float32)
            dst = np.array(
                [
                    [0.0, 0.0],
                    [calibration.real_length_m, 0.0],
                    [calibration.real_length_m, calibration.real_width_m],
                    [0.0, calibration.real_width_m],
                ],
                dtype=np.float32,
            )
            self._homography = cv2.getPerspectiveTransform(src, dst)

    @property
    def is_calibrated(self) -> bool:
        return self._homography is not None

    def to_world(self, pixel: tuple[float, float]) -> tuple[float, float]:
        """Map one pixel coordinate to ground-plane meters."""
        if self._homography is None:
            raise RuntimeError("Calibrator is not calibrated — provide a Calibration.")
        pts = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, self._homography)[0, 0]
        return (float(out[0]), float(out[1]))

    def world_distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """Euclidean distance in meters between two ground-plane points."""
        return float(np.hypot(b[0] - a[0], b[1] - a[1]))
