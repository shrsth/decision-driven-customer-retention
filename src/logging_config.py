"""Structured logging setup shared across the pipeline and services."""

import logging

_CONFIGURED = False


def get_logger(name: str = "retention") -> logging.Logger:
    """Return a logger, configuring a timestamped stdout handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
