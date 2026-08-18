"""Command-line interface for the speed-detector pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

from speed_detector import __version__
from speed_detector.config import Settings
from speed_detector.pipeline.source import FileSource
from speed_detector.utils.logging import configure_logging

app = typer.Typer(
    name="speed-detector",
    help="Real-time vehicle speed detection and traffic analytics.",
    no_args_is_help=True,
)


@app.command()
def run(
    source: Path = typer.Argument(..., help="Video file path or RTSP URL."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Open a video source and stream frames through the pipeline.

    Phase 1 milestone: reads the source and reports frame shape + FPS to prove
    the ingestion path works end-to-end. Detection, tracking, and speed
    estimation arrive in Phase 2.
    """
    configure_logging(verbose=verbose)
    settings = Settings()
    typer.echo(f"Detector backend: {settings.detector_backend}")

    video = FileSource(str(source))
    typer.echo(f"Source: {source}")
    typer.echo(f"Frame size (WxH): {video.frame_size[0]}x{video.frame_size[1]}")
    typer.echo(f"FPS: {video.fps:.2f}")

    count = 0
    for _frame in video:
        count += 1
    typer.echo(f"Read {count} frames.")


@app.command()
def version() -> None:
    """Print the installed version and exit."""
    typer.echo(f"speed-detector {__version__}")
