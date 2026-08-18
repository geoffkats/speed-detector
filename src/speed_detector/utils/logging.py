"""Structured logging configuration via ``structlog``."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(verbose: bool = False) -> None:
    """Configure stdlib logging + structlog processors.

    Args:
        verbose: when true, emit DEBUG-level logs.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
