"""Shared pytest fixtures for the speed-detector test suite."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Generate a tiny deterministic video (4 frames) for ingestion tests.

    Skips if the platform lacks a writable ``mp4v`` codec, so CI stays green
    on minimal OpenCV builds.
    """
    path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("mp4v codec unavailable — cannot synthesize test video")
    try:
        for i in range(4):
            frame = np.full((48, 64, 3), 40 + i * 50, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path
