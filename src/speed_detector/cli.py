"""Command-line interface for the speed-detector pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import typer

from speed_detector import __version__
from speed_detector.annotator import Annotator
from speed_detector.config import Settings
from speed_detector.pipeline.calibrator import Calibration, Calibrator
from speed_detector.pipeline.detector import UltralyticsDetector
from speed_detector.pipeline.pipeline import Pipeline
from speed_detector.pipeline.source import open_source
from speed_detector.pipeline.speed import SpeedEstimator
from speed_detector.pipeline.tracker import ByteTrackTracker
from speed_detector.utils.logging import configure_logging

app = typer.Typer(
    name="speed-detector",
    help="Real-time vehicle speed detection and traffic analytics.",
    no_args_is_help=True,
)


def _resolve_weights(settings: Settings, override: str | None) -> str:
    if override:
        return override
    if settings.detector_backend == "ultralytics":
        return "yolo11n.pt"
    if Path(settings.onnx_model_path).is_file():
        return settings.onnx_model_path
    typer.echo(
        f"ONNX model not found at {settings.onnx_model_path}; falling back to "
        "yolo11n.pt (run `uv run python -m scripts.export_model` to create the ONNX)."
    )
    return "yolo11n.pt"


def _load_calibration(path: Path) -> Calibration:
    data = json.loads(path.read_text(encoding="utf-8"))
    src = [tuple(float(v) for v in p) for p in data["src"]]
    if len(src) != 4:
        raise ValueError("Calibration 'src' must have exactly 4 (x, y) points.")
    return Calibration(
        src=tuple(src),  # type: ignore[arg-type]
        real_width_m=float(data["real_width_m"]),
        real_length_m=float(data["real_length_m"]),
    )


def _make_writer(path: Path, frame_size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, frame_size)


@app.command()
def run(
    source: str = typer.Argument(..., help="Video file, RTSP URL, or webcam index."),
    annotate: Path | None = typer.Option(
        None, "--annotate", "-a", help="Write the annotated video to this path."
    ),
    weights: str | None = typer.Option(
        None, "--weights", "-w", help="Model weights (.pt or .onnx)."
    ),
    calibration: Path | None = typer.Option(
        None, "--calibration", "-c", help="JSON file with 4-point calibration."
    ),
    max_frames: int | None = typer.Option(
        None, "--max-frames", help="Stop after this many frames."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run the detection pipeline over a video source and annotate frames."""
    configure_logging(verbose=verbose)
    settings = Settings()

    weights_path = _resolve_weights(settings, weights)
    typer.echo(f"Detector: {settings.detector_backend} ({weights_path})")

    video = open_source(source)
    typer.echo(
        f"Source: {source}  {video.frame_size[0]}x{video.frame_size[1]} @ {video.fps:.1f} fps"
    )

    calibrator = Calibrator(_load_calibration(calibration)) if calibration else Calibrator(None)
    if calibration:
        typer.echo("Calibration: loaded — speeds in real mph/kph.")
    else:
        typer.echo("Calibration: none — boxes/tracks only (no mph labels).")

    detector = UltralyticsDetector(
        weights=weights_path,
        confidence=settings.yolo_confidence,
        iou=settings.yolo_iou,
    )
    tracker = ByteTrackTracker(frame_rate=video.fps)
    speed_estimator = SpeedEstimator(calibrator, fps=video.fps)
    annotator = Annotator()
    pipeline = Pipeline(video, detector, tracker, calibrator, speed_estimator, annotator)

    writer = _make_writer(annotate, video.frame_size, video.fps) if annotate else None
    count = 0
    active_tracks = 0
    try:
        for result in pipeline.run(max_frames=max_frames):
            if writer is not None:
                writer.write(result.annotated)
            count += 1
            active_tracks = len(result.tracks)
    finally:
        video.close()
        if writer is not None:
            writer.release()

    typer.echo(f"Processed {count} frames; {active_tracks} tracks in the last frame.")
    if annotate:
        typer.echo(f"Annotated video written to: {annotate}")


@app.command()
def version() -> None:
    """Print the installed version and exit."""
    typer.echo(f"speed-detector {__version__}")
