"""Structured logging setup."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure structlog once for the CLI or API process."""
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
