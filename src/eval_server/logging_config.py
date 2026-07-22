"""Structured logging configuration — JSON or text format."""

import json
import logging
import sys
from datetime import datetime, timezone

EXTRA_FIELDS = (
    "request_id",
    "gpu_id",
    "task_name",
    "duration_ms",
    "success",
    "error_type",
    "queue_depth",
)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        for key in EXTRA_FIELDS:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(level: str = "INFO", log_format: str = "text"):
    """Configure structured or text logging."""
    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    logging.root.handlers = [handler]
    logging.root.setLevel(level)
