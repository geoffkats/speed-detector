"""Typed data records flowing through the pipeline.

These are the typed boundaries described in ``ARCHITECTURE.md §3``. Frozen
``slots=True`` dataclasses make the data flow cheap to copy and impossible to
mutate by accident — a property the legacy ``VehicleCounter`` lacked.

Note: ``BBox`` uses ``xyxy`` (top-left / bottom-right) to match the convention
used by Ultralytics and ``supervision`` throughout the CV stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class VehicleClass(IntEnum):
    """Vehicle categories the detector reports (internal ids, not COCO ids)."""

    CAR = 0
    TRUCK = 1
    BUS = 2
    MOTORCYCLE = 3
    BICYCLE = 4


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class Detection:
    """A single vehicle detection on one frame."""

    bbox: BBox
    confidence: float
    cls: VehicleClass


@dataclass(frozen=True, slots=True)
class Track:
    """A tracked vehicle identity at one frame."""

    id: int
    bbox: BBox
    confidence: float
    cls: VehicleClass
    frame_index: int


@dataclass(frozen=True, slots=True)
class SpeedReading:
    """A speed estimate for a tracked vehicle, in real-world units."""

    track_id: int
    cls: VehicleClass
    speed_mph: float
    speed_kph: float
    world_position_m: tuple[float, float]
    frame_index: int
