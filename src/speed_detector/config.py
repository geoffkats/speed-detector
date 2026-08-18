"""Application configuration loaded from environment / ``.env``.

See ``.env.example`` for every supported key and its default. All values have
sensible defaults, so ``.env`` is optional — the project runs out of the box.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the speed-detector pipeline and services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Detection ---
    detector_backend: Literal["onnx", "ultralytics"] = "onnx"
    onnx_model_path: str = "models/yolo11n.onnx"
    yolo_confidence: float = 0.35
    yolo_iou: float = 0.7

    # --- Tracking ---
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"

    # --- Storage ---
    db_url: str = "sqlite:///data/speed.db"

    # --- Streaming ---
    stream_max_fps: int = 15

    # --- Logging ---
    log_level: str = "INFO"

    # --- Local LLM (Phase 5) ---
    ollama_enabled: bool = False
    ollama_model: str = "qwen2.5:3b"
