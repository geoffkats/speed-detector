"""Smoke tests for the package skeleton and ingestion path."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from speed_detector import __version__
from speed_detector.cli import app
from speed_detector.config import Settings
from speed_detector.pipeline.source import FileSource
from speed_detector.types import BBox, Detection, VehicleClass

runner = CliRunner()


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.detector_backend == "onnx"
    assert settings.tracker == "bytetrack"
    assert settings.stream_max_fps == 15


def test_types_construct() -> None:
    bbox = BBox(x1=1.0, y1=2.0, x2=4.0, y2=6.0)
    detection = Detection(bbox=bbox, confidence=0.9, cls=VehicleClass.CAR)
    assert detection.cls == VehicleClass.CAR
    assert detection.confidence == pytest.approx(0.9)


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output


def test_filesource_reads_synthetic_video(synthetic_video: Path) -> None:
    source = FileSource(str(synthetic_video))
    assert source.frame_size == (64, 48)
    frames = list(source)
    assert len(frames) == 4
    assert frames[0].shape == (48, 64, 3)
